"""
evaluation/llm_judge.py
─────────────────────────────────────────────────────────────────────────────
LLM-as-judge evaluation for generated replies.

Evaluates each reply on 5 dimensions (1–5 rubric):
  1. Relevance
  2. Groundedness
  3. Helpfulness
  4. Brand_consistency
  5. Safety

The judge does NOT see the gold label (prevents leakage).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def judge_reply(
    generator,
    customer_message: str,
    draft_reply: str,
    retrieved_evidence: list[dict],
) -> Optional[dict[str, Any]]:
    """
    Run the LLM judge and return scores dict.
    Returns None if judge call fails.
    """
    return generator.judge(customer_message, draft_reply, retrieved_evidence)


def run_batch_evaluation(
    examples: list[dict],
    generator,
    results_dir: Path,
) -> list[dict]:
    """
    Run the LLM judge over a batch of examples.

    Each example should have:
      - customer_message
      - draft_reply
      - retrieved_evidence (list)

    Returns list of judge result dicts.
    """
    results = []
    failed = 0
    for i, ex in enumerate(examples):
        logger.info("Judging example %d/%d...", i + 1, len(examples))
        scores = judge_reply(
            generator=generator,
            customer_message=ex.get("customer_message", ""),
            draft_reply=ex.get("draft_reply", ""),
            retrieved_evidence=ex.get("retrieved_evidence", []),
        )
        if scores is None:
            failed += 1
            scores = {
                "relevance": 0, "groundedness": 0, "helpfulness": 0,
                "brand_consistency": 0, "safety": 0, "overall": 0,
                "reasoning": "JUDGE_FAILED",
            }
        result = {**ex, "judge_scores": scores}
        results.append(result)

    logger.info("Judge complete. %d succeeded, %d failed", len(results) - failed, failed)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "llm_judge_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Judge results saved to %s", out_path)

    # Print summary
    dims = ["relevance", "groundedness", "helpfulness", "brand_consistency", "safety", "overall"]
    print("\n" + "=" * 60)
    print("LLM Judge Summary (1–5 scale)")
    print("-" * 60)
    for dim in dims:
        scores_list = [r["judge_scores"].get(dim, 0) for r in results if r["judge_scores"].get(dim, 0) > 0]
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            print(f"  {dim:<22}: {avg:.2f}/5.0  (n={len(scores_list)})")
    print("=" * 60)

    return results
