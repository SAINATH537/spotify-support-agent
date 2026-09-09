#!/usr/bin/env python
"""
scripts/reproduce_results.py
─────────────────────────────────────────────────────────────────────────────
Single-entry-point reproduction script.

Runs the full evaluation pipeline in order:
  1. Intent classification evaluation (all 3 classifiers)
  2. Agent on golden set (or test split)
  3. Escalation evaluation + threshold sweep
  4. LLM judge scoring
  5. Human agreement template generation
  6. Failure analysis
  7. Print summary table

Target: < 15 minutes on --sample mode.

Usage:
  python scripts/reproduce_results.py                    # full pipeline
  python scripts/reproduce_results.py --sample 200      # fast (no LLM calls)
  python scripts/reproduce_results.py --no-llm          # skip LLM gen
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logging.getLogger("reproduce_results")
    t_start = time.perf_counter()

    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = ROOT / cfg["data"]["processed_dir"]
    golden_dir = ROOT / cfg["data"]["golden_dir"]
    results_dir = ROOT / cfg["evaluation"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Check prerequisites ───────────────────────────────────────────────────
    interactions_path = processed_dir / "spotify_interactions.csv"
    if not interactions_path.exists():
        logger.error(
            "Missing: %s\nRun first:\n"
            "  python scripts/prepare_data.py --sample 5000\n"
            "  python scripts/build_index.py --sample 3000",
            interactions_path
        )
        sys.exit(1)

    # ── Step 1: Intent evaluation ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Intent Classification Evaluation")
    logger.info("=" * 60)
    from evaluation.evaluate_intents import main as eval_intents, setup_logging as el
    import argparse as _ap
    eval_intents(_ap.Namespace(quick=False))

    # ── Step 2: Run agent on sample of test examples ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Running agent on test examples")
    logger.info("=" * 60)

    df = pd.read_csv(interactions_path)
    from src.intent.taxonomy import IntentTaxonomy
    taxonomy = IntentTaxonomy(config_path)
    seed = cfg.get("random_seed", 42)
    if "weak_intent" not in df.columns:
        df["weak_intent"] = taxonomy.weak_label_batch(df["customer_message"].tolist())

    n_eval = args.sample or 100
    test_examples = df.sample(n=min(n_eval, len(df)), random_state=seed).to_dict("records")
    logger.info("Running agent on %d examples...", len(test_examples))

    agent_results = []
    if not args.no_llm:
        from src.agent import SpotifySupportAgent
        agent = SpotifySupportAgent(config_path=config_path)
        for i, ex in enumerate(test_examples):
            if i % 10 == 0:
                logger.info("  Processing %d/%d...", i, len(test_examples))
            try:
                res = agent.run(ex["customer_message"])
                res["example_id"] = f"ex_{i:04d}"
                res["gold_intent"] = ex.get("weak_intent", "")
                res["should_escalate"] = False  # no gold escalation labels yet
                agent_results.append(res)
            except Exception as e:
                logger.warning("Agent failed on example %d: %s", i, e)
    else:
        logger.info("Skipping LLM generation (--no-llm). Using stub for escalation eval.")
        from src.intent.classifier import SentenceTransformerClassifier
        from src.escalation.policy import EscalationPolicy
        clf_path = ROOT / cfg["classification"]["model_path"]
        policy = EscalationPolicy(config_path)
        if clf_path.exists():
            clf = SentenceTransformerClassifier.load(clf_path)
            for i, ex in enumerate(test_examples):
                intent, conf = clf.predict_one(ex["customer_message"])
                esc = policy.decide(intent, conf, [], 0.5, False)
                agent_results.append({
                    "example_id": f"ex_{i:04d}",
                    "intent": intent,
                    "intent_confidence": round(conf, 4),
                    "retrieved_evidence": [],
                    "draft_reply": "",
                    "grounding_score": 0.5,
                    "decision": esc.decision,
                    "decision_reason": esc.decision_reason,
                    "risk_flags": esc.risk_flags,
                    "gold_intent": ex.get("weak_intent", ""),
                    "should_escalate": False,
                })

    agent_out = results_dir / "agent_results.json"
    with open(agent_out, "w") as f:
        json.dump(agent_results, f, indent=2, ensure_ascii=False)
    logger.info("Agent results saved: %s", agent_out)

    # ── Step 3: Escalation evaluation + threshold sweep ───────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Escalation Evaluation")
    logger.info("=" * 60)
    from evaluation.evaluate_escalation import threshold_sweep, compute_escalation_metrics
    from src.escalation.policy import EscalationPolicy
    policy = EscalationPolicy(config_path)
    thresholds = cfg["evaluation"]["automation_coverage_thresholds"]
    sweep_df = threshold_sweep(agent_results, policy, thresholds, results_dir)

    # ── Step 4: LLM judge ─────────────────────────────────────────────────────
    if not args.no_llm and agent_results:
        logger.info("=" * 60)
        logger.info("STEP 4: LLM Judge Evaluation")
        logger.info("=" * 60)
        from src.generation.generator import ResponseGenerator
        gen_cfg = cfg["generation"]
        generator = ResponseGenerator(
            provider=gen_cfg.get("provider", "stub"),
            model=gen_cfg.get("model", "gemini-1.5-flash"),
            temperature=0.0,  # deterministic judge
        )
        from evaluation.llm_judge import run_batch_evaluation
        judge_sample = agent_results[:min(50, len(agent_results))]
        run_batch_evaluation(judge_sample, generator, results_dir)

    # ── Step 5: Human agreement template ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Human Agreement Template")
    logger.info("=" * 60)
    from evaluation.human_agreement import generate_human_score_template, run_agreement_analysis
    llm_judge_path = results_dir / "llm_judge_results.json"
    if llm_judge_path.exists():
        with open(llm_judge_path) as f:
            judge_results = json.load(f)
        generate_human_score_template(judge_results, n=50, results_dir=results_dir, seed=seed)
    run_agreement_analysis(results_dir)

    # ── Step 6: Failure analysis ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Failure Analysis")
    logger.info("=" * 60)
    _run_failure_analysis(agent_results, results_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.perf_counter() - t_start) / 60
    logger.info("=" * 60)
    logger.info("REPRODUCE COMPLETE in %.1f minutes", elapsed)
    logger.info("Results in: %s", results_dir)
    logger.info("=" * 60)
    _print_summary(results_dir)


def _run_failure_analysis(agent_results: list[dict], results_dir: Path) -> None:
    """Categorise and save failure cases."""
    failures = []
    for ex in agent_results:
        fail_reasons = []
        if ex.get("intent_confidence", 1.0) < 0.4:
            fail_reasons.append("low_confidence")
        if not ex.get("retrieved_evidence"):
            fail_reasons.append("no_retrieval")
        if ex.get("grounding_score", 1.0) < 0.3:
            fail_reasons.append("low_grounding")
        if ex.get("generation_failed", False):
            fail_reasons.append("generation_failed")

        if fail_reasons:
            failures.append({
                "example_id": ex.get("example_id"),
                "customer_message": ex.get("customer_message", "")[:200],
                "intent": ex.get("intent"),
                "intent_confidence": ex.get("intent_confidence"),
                "decision": ex.get("decision"),
                "failure_categories": fail_reasons,
                "draft_reply": ex.get("draft_reply", "")[:200],
            })

    out = results_dir / "failure_cases.json"
    with open(out, "w") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    logging.getLogger("reproduce_results").info(
        "Failure analysis: %d failures saved to %s", len(failures), out
    )


def _print_summary(results_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    intent_path = results_dir / "intent_evaluation.json"
    if intent_path.exists():
        with open(intent_path) as f:
            intent_results = json.load(f)
        print("\nIntent Classification:")
        print(f"  {'Classifier':<30} {'Accuracy':>10} {'Macro F1':>10}")
        print(f"  {'-'*53}")
        for r in intent_results:
            print(f"  {r['classifier']:<30} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")

    sweep_path = results_dir / "escalation_threshold_sweep.csv"
    if sweep_path.exists():
        import pandas as pd
        df = pd.read_csv(sweep_path)
        print("\nEscalation Threshold Sweep:")
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Unsafe Rate':>12} {'Esc F1':>8}")
        for _, row in df.iterrows():
            print(f"  {row['threshold']:>10.2f} {row['automation_coverage']:>10.4f} "
                  f"{row['unsafe_auto_handle_rate']:>12.4f} {row['escalation_f1']:>8.4f}")

    print("=" * 70)
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce all evaluation results")
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of test examples to run agent on (default: 100)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM generation (faster, no API key needed)")
    args = parser.parse_args()
    main(args)
