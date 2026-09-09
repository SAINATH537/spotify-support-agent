"""
evaluation/evaluate_replies.py
─────────────────────────────────────────────────────────────────────────────
Compare three reply systems using the LLM judge:
  1. Nearest-neighbour (no generation — just return best historical reply)
  2. LLM without retrieval (no RAG)
  3. LLM with retrieval (full RAG system)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
logger = logging.getLogger(__name__)


def evaluate_three_systems(
    test_examples: list[dict],
    agent,
    generator,
    faiss_store,
    results_dir: Path,
) -> None:
    """
    Run all three reply systems and score with LLM judge.
    test_examples: list of {customer_message, ...}
    """
    from evaluation.llm_judge import run_batch_evaluation
    from src.retrieval.embeddings import embed_single

    systems = {
        "nearest_neighbour": [],
        "llm_no_rag": [],
        "llm_with_rag": [],
    }

    for ex in test_examples:
        msg = ex.get("customer_message", "")
        qvec = embed_single(msg)
        retrieved = faiss_store.search(qvec, top_k=5)

        # System 1: Nearest-neighbour (just copy best historical reply)
        nn_reply = retrieved[0]["brand_response"] if retrieved else "Please contact support."
        systems["nearest_neighbour"].append({
            "customer_message": msg,
            "draft_reply": nn_reply,
            "retrieved_evidence": retrieved,
            "example_id": ex.get("example_id", ""),
        })

        # System 2: LLM without retrieval
        no_rag = generator.generate(
            brand="SpotifyCares", customer_message=msg,
            intent=ex.get("intent", "Other / Ambiguous"),
            intent_description="",
            retrieved_evidence=[],
        )
        systems["llm_no_rag"].append({
            "customer_message": msg,
            "draft_reply": no_rag["draft_reply"],
            "retrieved_evidence": [],
            "example_id": ex.get("example_id", ""),
        })

        # System 3: LLM with retrieval (full system)
        with_rag = generator.generate(
            brand="SpotifyCares", customer_message=msg,
            intent=ex.get("intent", "Other / Ambiguous"),
            intent_description="",
            retrieved_evidence=retrieved,
        )
        systems["llm_with_rag"].append({
            "customer_message": msg,
            "draft_reply": with_rag["draft_reply"],
            "retrieved_evidence": retrieved,
            "example_id": ex.get("example_id", ""),
        })

    # Judge all systems
    for system_name, examples in systems.items():
        logger.info("Judging system: %s", system_name)
        judge_results = run_batch_evaluation(examples, generator, results_dir)
        # Save per-system
        out = results_dir / f"reply_eval_{system_name}.json"
        with open(out, "w") as f:
            json.dump(judge_results, f, indent=2)

    logger.info("Reply evaluation complete.")
