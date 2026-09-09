"""
evaluation/evaluate_escalation.py
─────────────────────────────────────────────────────────────────────────────
Escalation policy evaluation with threshold sweep.

Metrics:
  - Escalation precision, recall, F1
  - Automation coverage
  - Unsafe auto-handle rate
  - Threshold sweep table + plot

Unsafe auto-handle = escalate=True in gold but AUTO_HANDLE predicted.
This is the most dangerous error — higher priority than false escalations.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def compute_escalation_metrics(
    y_true_esc: list[bool],
    y_pred_esc: list[bool],
) -> dict:
    """Compute precision, recall, F1 for escalation decisions."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p, r, f, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_esc, pos_label=True, average="binary", zero_division=0
    )
    n_auto = sum(1 for p in y_pred_esc if not p)
    n_total = len(y_pred_esc)
    coverage = n_auto / n_total if n_total else 0.0
    unsafe = sum(1 for true, pred in zip(y_true_esc, y_pred_esc)
                 if true and not pred)  # should escalate but auto-handled
    unsafe_rate = unsafe / n_auto if n_auto > 0 else 0.0

    return {
        "escalation_precision": round(float(p), 4),
        "escalation_recall": round(float(r), 4),
        "escalation_f1": round(float(f), 4),
        "automation_coverage": round(coverage, 4),
        "unsafe_auto_handle_rate": round(unsafe_rate, 4),
        "n_total": n_total,
        "n_auto_handled": n_auto,
        "n_escalated": n_total - n_auto,
        "n_unsafe": unsafe,
    }


def threshold_sweep(
    examples: list[dict],
    policy,
    thresholds: list[float],
    results_dir: Path,
) -> pd.DataFrame:
    """
    Sweep confidence threshold and compute automation coverage vs unsafe rate.
    """
    rows = []
    for thresh in thresholds:
        preds = []
        for ex in examples:
            esc_result = policy.decide_with_overrides(
                intent=ex.get("intent", "Other / Ambiguous"),
                intent_confidence=ex.get("intent_confidence", 0.5),
                retrieved_evidence=ex.get("retrieved_evidence", []),
                grounding_score=ex.get("grounding_score", 0.5),
                generation_failed=ex.get("generation_failed", False),
                min_intent_confidence=thresh,
            )
            preds.append(esc_result.decision == "ESCALATE")
        y_true = [bool(ex.get("should_escalate", False)) for ex in examples]
        metrics = compute_escalation_metrics(y_true, preds)
        metrics["threshold"] = thresh
        rows.append(metrics)

    sweep_df = pd.DataFrame(rows)
    sweep_path = results_dir / "escalation_threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    logger.info("Threshold sweep saved: %s", sweep_path)

    # Plot
    _plot_sweep(sweep_df, results_dir)
    return sweep_df


def _plot_sweep(df: pd.DataFrame, results_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["automation_coverage"], df["unsafe_auto_handle_rate"],
                "o-", color="#1DB954", linewidth=2, markersize=8, label="Operating points")
        for _, row in df.iterrows():
            ax.annotate(
                f"t={row['threshold']:.2f}",
                (row["automation_coverage"], row["unsafe_auto_handle_rate"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8
            )
        ax.set_xlabel("Automation Coverage (fraction auto-handled)", fontsize=12)
        ax.set_ylabel("Unsafe Auto-Handle Rate", fontsize=12)
        ax.set_title("Automation Coverage vs Unsafe Auto-Handle Rate\n(Threshold Sweep)", fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "escalation_threshold_sweep.png", dpi=150)
        plt.close()
        logger.info("Threshold sweep plot saved")
    except Exception as e:
        logger.warning("Could not plot threshold sweep: %s", e)
