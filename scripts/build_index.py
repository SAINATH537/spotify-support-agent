#!/usr/bin/env python
"""
scripts/build_index.py
─────────────────────────────────────────────────────────────────────────────
Phase 6: Build the FAISS retrieval index from processed interactions.

Steps:
  1. Load retrieval-worthy interactions from processed CSV
  2. Exclude golden test set tweet IDs (leakage prevention)
  3. Train the three intent classifiers on weakly-labelled data
  4. Embed customer messages with sentence-transformer
  5. Build and save FAISS index
  6. Train and save classifiers

Usage:
  python scripts/build_index.py
  python scripts/build_index.py --sample 3000   # fast mode
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logging.getLogger("build_index")

    config_path = ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    processed_dir = ROOT / cfg["data"]["processed_dir"]
    golden_dir = ROOT / cfg["data"]["golden_dir"]
    embed_model = cfg["embedding"]["model_name"]
    embed_batch = cfg["embedding"]["batch_size"]
    cache_dir = ROOT / cfg["embedding"]["cache_dir"]
    index_path = ROOT / cfg["retrieval"]["index_path"]
    classifier_path = ROOT / cfg["classification"]["model_path"]
    test_split = cfg["classification"]["test_split"]
    dev_split = cfg["classification"]["dev_split"]

    logger.info("=" * 60)
    logger.info("PHASE 6: Building FAISS Index + Training Classifiers")
    logger.info("=" * 60)

    # ── Load interactions ────────────────────────────────────────────────────
    interactions_path = processed_dir / "spotify_interactions.csv"
    if not interactions_path.exists():
        logger.error(
            "Interactions file not found: %s\n"
            "Run: python scripts/prepare_data.py first.", interactions_path
        )
        sys.exit(1)

    df = pd.read_csv(interactions_path)
    logger.info("Loaded %d retrieval-worthy interactions", len(df))

    # ── Exclude golden test set (leakage prevention) ──────────────────────────
    golden_path = golden_dir / "spotify_golden_set_annotation_template_250.csv"
    excluded_tweet_ids: set[str] = set()
    if golden_path.exists():
        golden_df = pd.read_csv(golden_path)
        excluded_tweet_ids = set(golden_df["tweet_id"].astype(str).tolist())
        before = len(df)
        df = df[~df["customer_tweet_id"].astype(str).isin(excluded_tweet_ids)].copy()
        logger.info(
            "Excluded %d golden test examples from index (leakage prevention)",
            before - len(df),
        )

    # ── Optional sample for fast mode ────────────────────────────────────────
    if args.sample and args.sample > 0:
        logger.warning("SAMPLE MODE: restricting to %d interactions", args.sample)
        df = df.sample(n=min(args.sample, len(df)), random_state=seed)

    # ── Load taxonomy and get weak labels ─────────────────────────────────────
    from src.intent.taxonomy import IntentTaxonomy
    taxonomy = IntentTaxonomy(config_path)

    if "weak_intent" not in df.columns:
        logger.info("Computing weak intent labels...")
        df["weak_intent"] = taxonomy.weak_label_batch(df["customer_message"].tolist())

    # ── Train/dev/test split ─────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    texts = df["customer_message"].tolist()
    labels = df["weak_intent"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_split, random_state=seed, stratify=labels
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=dev_split / (1 - test_split), random_state=seed, stratify=y_train
    )
    logger.info("Split: train=%d, dev=%d, test=%d", len(X_train), len(X_dev), len(X_test))

    # Save split indices for reproducibility
    split_path = processed_dir / "classifier_split.json"
    with open(split_path, "w") as f:
        json.dump({
            "train_size": len(X_train), "dev_size": len(X_dev), "test_size": len(X_test),
            "seed": seed, "test_split": test_split, "dev_split": dev_split,
        }, f)
    logger.info("Split info saved to %s", split_path)

    # ── Train classifiers ─────────────────────────────────────────────────────
    from src.intent.baseline import MajorityClassifier, TFIDFLogisticRegression
    from src.intent.classifier import SentenceTransformerClassifier

    logger.info("Training Majority classifier...")
    maj = MajorityClassifier()
    maj.fit(X_train, y_train)
    maj.save(processed_dir / "majority_clf.joblib")

    logger.info("Training TF-IDF LR classifier...")
    tfidf_lr = TFIDFLogisticRegression(random_state=seed)
    tfidf_lr.fit(X_train, y_train)
    tfidf_lr.save(processed_dir / "tfidf_lr_clf.joblib")

    logger.info("Training Sentence-Transformer classifier...")
    st_clf = SentenceTransformerClassifier(
        model_name=embed_model,
        cache_dir=str(cache_dir),
        random_state=seed,
    )
    st_clf.fit(X_train, y_train)
    st_clf.save(classifier_path)

    logger.info("All classifiers saved.")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    logger.info("Building FAISS index from %d interactions...", len(df))
    from src.retrieval.embeddings import embed_texts
    from src.retrieval.faiss_store import FAISSStore

    # Embed ALL interactions (not just train split — retrieval uses full pool)
    embeddings = embed_texts(
        df["customer_message"].tolist(),
        model_name=embed_model,
        batch_size=embed_batch,
        cache_dir=str(cache_dir),
        cache_key_prefix="retrieval_",
    )

    # Predict intents for all interactions for intent-aware retrieval
    logger.info("Predicting intents for retrieval index...")
    predicted_intents = st_clf.predict(df["customer_message"].tolist())
    df["intent"] = predicted_intents

    metadata = df[[
        "interaction_id", "customer_message", "brand_response",
        "resolution_type", "intent", "customer_tweet_id", "brand_tweet_id", "timestamp"
    ]].to_dict("records")

    store = FAISSStore()
    store.build(embeddings, metadata, intent_key="intent")
    store.save(index_path)

    logger.info("=" * 60)
    logger.info("PHASE 6 COMPLETE")
    logger.info("FAISS index: %d vectors | Classifiers: 3 saved", store.size)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index and train classifiers")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only N interactions (fast mode)")
    args = parser.parse_args()
    main(args)
