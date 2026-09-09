"""
evaluation/evaluate_intents.py
─────────────────────────────────────────────────────────────────────────────
Intent classification evaluation: all three classifiers vs golden test set.

Metrics:
  - Accuracy
  - Macro F1 (headline metric — handles class imbalance)
  - Weighted F1
  - Per-class precision, recall, F1
  - Confusion matrix (saved as image)

Usage:
  python evaluation/evaluate_intents.py
  python evaluation/evaluate_intents.py --quick   # evaluate on interactions test split only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def evaluate_classifier(clf, X_test: list[str], y_test: list[str], name: str) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    y_pred = clf.predict(X_test)
    labels = sorted(set(y_test) | set(y_pred))

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    p, r, f, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        labels[i]: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "classifier": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_test": len(y_test),
    }


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 9))
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=[l[:20] for l in labels],
        yticklabels=[l[:20] for l in labels],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.getLogger("evaluate_intents").info("Confusion matrix saved: %s", out_path)


def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logging.getLogger("evaluate_intents")

    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = ROOT / cfg["data"]["processed_dir"]
    golden_dir = ROOT / cfg["data"]["golden_dir"]
    results_dir = ROOT / cfg["evaluation"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test data ────────────────────────────────────────────────────────
    from src.intent.baseline import MajorityClassifier, TFIDFLogisticRegression
    from src.intent.classifier import SentenceTransformerClassifier
    from src.intent.taxonomy import IntentTaxonomy

    taxonomy = IntentTaxonomy(config_path)

    # Use golden set if it has verified labels, else fall back to interactions test split
    golden_path = golden_dir / "spotify_golden_set_annotation_template_250.csv"
    split_path = processed_dir / "classifier_split.json"

    # Load interactions for test split evaluation
    interactions_path = processed_dir / "spotify_interactions.csv"
    if not interactions_path.exists():
        logger.error("Run prepare_data.py and build_index.py first.")
        sys.exit(1)

    df = pd.read_csv(interactions_path)
    if "weak_intent" not in df.columns:
        df["weak_intent"] = taxonomy.weak_label_batch(df["customer_message"].tolist())

    seed = cfg.get("random_seed", 42)
    test_split = cfg["classification"]["test_split"]
    dev_split = cfg["classification"]["dev_split"]

    from sklearn.model_selection import train_test_split
    texts = df["customer_message"].tolist()
    labels = df["weak_intent"].tolist()
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=test_split, random_state=seed, stratify=labels
    )

    logger.info("Evaluating on %d test examples", len(X_test))

    # ── Load classifiers ──────────────────────────────────────────────────────
    all_results = []

    maj_path = processed_dir / "majority_clf.joblib"
    if maj_path.exists():
        maj = MajorityClassifier.load(maj_path)
        res = evaluate_classifier(maj, X_test, y_test, "MajorityClassifier")
        all_results.append(res)
        logger.info("Majority: accuracy=%.4f  macro_f1=%.4f", res["accuracy"], res["macro_f1"])

    tfidf_path = processed_dir / "tfidf_lr_clf.joblib"
    if tfidf_path.exists():
        tfidf = TFIDFLogisticRegression.load(tfidf_path)
        res = evaluate_classifier(tfidf, X_test, y_test, "TF-IDF LR")
        all_results.append(res)
        logger.info("TF-IDF LR: accuracy=%.4f  macro_f1=%.4f", res["accuracy"], res["macro_f1"])

    st_path = ROOT / cfg["classification"]["model_path"]
    if st_path.exists():
        st_clf = SentenceTransformerClassifier.load(st_path)
        res = evaluate_classifier(st_clf, X_test, y_test, "SentenceTransformer LR")
        all_results.append(res)
        logger.info("ST LR: accuracy=%.4f  macro_f1=%.4f", res["accuracy"], res["macro_f1"])

    if not all_results:
        logger.error("No classifiers found. Run build_index.py first.")
        sys.exit(1)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = results_dir / "intent_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved: %s", out_path)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Classifier':<30} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 70)
    for res in all_results:
        print(f"{res['classifier']:<30} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} {res['weighted_f1']:>12.4f}")
    print("=" * 70)

    # ── Confusion matrices ────────────────────────────────────────────────────
    for res in all_results:
        name_clean = res["classifier"].replace(" ", "_").replace("/", "_")
        plot_confusion_matrix(
            res["confusion_matrix"],
            res["labels"],
            f"Confusion Matrix — {res['classifier']}",
            results_dir / f"cm_{name_clean}.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args())
