"""
evaluation/human_agreement.py
─────────────────────────────────────────────────────────────────────────────
Validates the LLM judge by comparing to human scores on 50 examples.

Computes:
  - Spearman correlation per dimension
  - Cohen's kappa for overall binary decision (good >= 3 / poor < 3)

Usage: fill in human scores in results/human_scores_50.csv, then run.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
logger = logging.getLogger(__name__)

DIMENSIONS = ["relevance", "groundedness", "helpfulness", "brand_consistency", "safety", "overall"]


def compute_agreement(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> dict:
    """
    Args:
        human_df: columns = ['example_id'] + DIMENSIONS (human scores)
        llm_df:   columns = ['example_id'] + DIMENSIONS (LLM judge scores)
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    merged = human_df.merge(llm_df, on="example_id", suffixes=("_human", "_llm"))

    results = {}
    for dim in DIMENSIONS:
        hcol = f"{dim}_human"
        lcol = f"{dim}_llm"
        if hcol not in merged.columns or lcol not in merged.columns:
            continue
        h = merged[hcol].values
        l = merged[lcol].values
        corr, p = spearmanr(h, l)
        results[f"{dim}_spearman_r"] = round(float(corr), 4)
        results[f"{dim}_spearman_p"] = round(float(p), 4)

    # Cohen's kappa on binarised overall (>=3 = positive)
    h_bin = (merged["overall_human"] >= 3).astype(int).values
    l_bin = (merged["overall_llm"] >= 3).astype(int).values
    kappa = cohen_kappa_score(h_bin, l_bin)
    results["overall_kappa"] = round(float(kappa), 4)

    return results


def generate_human_score_template(
    llm_results: list[dict], n: int, results_dir: Path, seed: int = 42
) -> None:
    """Generate a CSV template for human annotators to fill in scores."""
    import random
    random.seed(seed)
    sample = random.sample(llm_results, min(n, len(llm_results)))
    rows = []
    for ex in sample:
        rows.append({
            "example_id": ex.get("example_id", ""),
            "customer_message": ex.get("customer_message", ""),
            "draft_reply": ex.get("draft_reply", ""),
            **{dim: "" for dim in DIMENSIONS},
            "notes": "",
        })
    df = pd.DataFrame(rows)
    out = results_dir / "human_scores_50_template.csv"
    df.to_csv(out, index=False)
    logger.info("Human scoring template saved: %s", out)
    print(f"\nHuman scoring template created at: {out}")
    print("Fill in the score columns (1–5) for each example, save as human_scores_50.csv")


def run_agreement_analysis(results_dir: Path) -> None:
    """Run agreement analysis if human scores are available."""
    human_path = results_dir / "human_scores_50.csv"
    llm_path = results_dir / "llm_judge_results.json"

    if not human_path.exists():
        logger.warning("Human scores not found at %s. Skipping agreement analysis.", human_path)
        return

    if not llm_path.exists():
        logger.warning("LLM judge results not found. Run evaluate_replies.py first.")
        return

    human_df = pd.read_csv(human_path)
    with open(llm_path) as f:
        llm_results = json.load(f)

    llm_df = pd.DataFrame([
        {"example_id": r.get("example_id", ""), **r.get("judge_scores", {})}
        for r in llm_results
    ])

    agreement = compute_agreement(human_df, llm_df)
    out_path = results_dir / "human_agreement.json"
    with open(out_path, "w") as f:
        json.dump(agreement, f, indent=2)

    print("\n" + "=" * 50)
    print("Human–LLM Judge Agreement")
    print("-" * 50)
    for k, v in agreement.items():
        print(f"  {k:<35}: {v}")
    print("=" * 50)
    logger.info("Agreement saved to %s", out_path)
