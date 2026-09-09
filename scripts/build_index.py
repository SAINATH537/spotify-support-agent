#!/usr/bin/env python
"""
scripts/build_index.py
─────────────────────────────────────────────────────────────────────────────
Build the FAISS retrieval index and train all intent classifiers.

Steps:
  1. Load retrieval-worthy interactions from processed CSV
  2. Exclude ALL golden set tweet IDs (leakage prevention — CRITICAL)
  3. Train three intent classifiers on weakly-labelled data
  4. Embed customer messages with sentence-transformer (cached)
  5. Predict intents for index metadata (intent-aware retrieval)
  6. Build and save FAISS index with full metadata
  7. Save classifier split info for reproducibility

Golden exclusion: uses customer_tweet_id from the interactions CSV.
  The golden template is drawn from the same pool — so we must exclude
  by tweet_id at index build time, not just filter by file existence.

Usage:
  python scripts/build_index.py           # full run
  python scripts/build_index.py --skip-classifier  # index only (fast)

Data-split note:
  The classifier train/test split is currently TWEET-LEVEL, not
  conversation-level. This is documented as a known limitation.
  Conversation-level splitting will be implemented in the next phase.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
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


def load_golden_tweet_ids(golden_dir: Path) -> set[str]:
    """
    Load ALL tweet IDs from the golden annotation template.
    Even though labels are currently weak / not manually verified,
    these examples MUST be excluded from the retrieval index to prevent
    evaluation leakage.

    Returns a set of customer_tweet_id strings.
    """
    # The golden template produced by prepare_data.py
    golden_template = golden_dir / "spotify_golden_set_annotation_template_250.csv"
    # Future hand-labelled golden set (may not exist yet)
    golden_final = golden_dir / "spotify_golden_set_250.csv"

    all_ids: set[str] = set()

    for path in [golden_template, golden_final]:
        if path.exists():
            gdf = pd.read_csv(path)
            if "tweet_id" in gdf.columns:
                ids = set(gdf["tweet_id"].astype(str).tolist())
                all_ids |= ids
                logger_global.info(
                    "  Golden exclusion: loaded %d IDs from %s", len(ids), path.name
                )

    logger_global.info("Total golden IDs to exclude: %d", len(all_ids))
    return all_ids


logger_global = logging.getLogger("build_index")


def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logger_global
    t_start = time.perf_counter()

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
    logger.info("Building FAISS Index + Training Classifiers")
    logger.info("Embedding model: %s", embed_model)
    logger.info("=" * 60)

    # ── Step 1: Load interactions ─────────────────────────────────────────────
    interactions_path = processed_dir / "spotify_interactions.csv"
    if not interactions_path.exists():
        logger.error(
            "Interactions file not found: %s\n"
            "Run: python scripts/prepare_data.py first.", interactions_path
        )
        sys.exit(1)

    df = pd.read_csv(interactions_path)
    logger.info("Loaded %d retrieval-worthy interactions from %s", len(df), interactions_path)

    # Verify schema — catch issues early
    required_cols = {
        "interaction_id", "conversation_id", "customer_message",
        "brand_response", "resolution_type", "customer_tweet_id",
        "brand_tweet_id", "timestamp",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Interactions CSV missing columns: %s", missing)
        sys.exit(1)

    # ── Step 2: Golden set exclusion (CRITICAL) ───────────────────────────────
    logger.info("Step 2: Excluding golden set tweet IDs from retrieval index...")
    golden_tweet_ids = load_golden_tweet_ids(golden_dir)
    before = len(df)
    df = df[~df["customer_tweet_id"].astype(str).isin(golden_tweet_ids)].copy()
    n_excluded = before - len(df)
    logger.info(
        "Excluded %d golden examples from index (%d remaining for retrieval)",
        n_excluded, len(df),
    )

    if len(df) == 0:
        logger.error("All interactions excluded! Check golden_dir path.")
        sys.exit(1)

    # Verify exclusion is clean — double-check intersection is empty
    remaining_ids = set(df["customer_tweet_id"].astype(str))
    overlap = remaining_ids & golden_tweet_ids
    assert len(overlap) == 0, f"BUG: {len(overlap)} golden IDs still in index after exclusion"
    logger.info("Golden exclusion verified: intersection of golden IDs and index IDs = 0")

    # ── Step 3: Load taxonomy and get weak labels ─────────────────────────────
    from src.intent.taxonomy import IntentTaxonomy
    taxonomy = IntentTaxonomy(config_path)

    if "weak_intent" not in df.columns:
        logger.info("Computing weak intent labels...")
        df["weak_intent"] = taxonomy.weak_label_batch(df["customer_message"].tolist())

    # ── Step 4: Train/dev/test split (for classifiers) ───────────────────────
    # NOTE: This split is TWEET-LEVEL (known limitation).
    # Conversation-level splitting is scheduled for the next phase.
    from sklearn.model_selection import train_test_split

    texts = df["customer_message"].tolist()
    labels = df["weak_intent"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_split, random_state=seed, stratify=labels
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train,
        test_size=dev_split / (1 - test_split),
        random_state=seed, stratify=y_train
    )
    logger.info("Split: train=%d, dev=%d, test=%d", len(X_train), len(X_dev), len(X_test))
    logger.warning(
        "DATA SPLIT WARNING: split is tweet-level, not conversation-level. "
        "Classifier evaluation may overestimate performance."
    )

    split_path = processed_dir / "classifier_split.json"
    with open(split_path, "w") as f:
        json.dump({
            "train_size": len(X_train), "dev_size": len(X_dev), "test_size": len(X_test),
            "seed": seed, "test_split": test_split, "dev_split": dev_split,
            "split_level": "tweet",  # documented limitation
            "note": "Conversation-level split not yet implemented",
        }, f, indent=2)
    logger.info("Split info saved to %s", split_path)

    # ── Step 5: Train classifiers ─────────────────────────────────────────────
    if not args.skip_classifier:
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
    else:
        logger.info("--skip-classifier: loading existing classifier for intent prediction...")
        from src.intent.classifier import SentenceTransformerClassifier
        if not Path(classifier_path).exists():
            logger.error(
                "Classifier not found at %s. Run without --skip-classifier first.", classifier_path
            )
            sys.exit(1)
        st_clf = SentenceTransformerClassifier.load(classifier_path)

    # ── Step 6: Embed and predict intents ─────────────────────────────────────
    logger.info("Step 6: Embedding %d customer messages...", len(df))
    from src.retrieval.embeddings import embed_texts
    from src.retrieval.faiss_store import FAISSStore

    embeddings = embed_texts(
        df["customer_message"].tolist(),
        model_name=embed_model,
        batch_size=embed_batch,
        cache_dir=str(cache_dir),
        cache_key_prefix="retrieval_",
    )

    logger.info("Embedding shape: %s", embeddings.shape)
    assert embeddings.shape[0] == len(df), "Embedding count must match interaction count"
    assert embeddings.dtype == np.float32

    logger.info("Predicting intents for retrieval index metadata...")
    predicted_intents = st_clf.predict(df["customer_message"].tolist())
    df = df.copy()
    df["intent"] = predicted_intents

    # ── Step 7: Build FAISS index ─────────────────────────────────────────────
    # Metadata retains all fields needed to trace evidence back to source data
    metadata = df[[
        "interaction_id",
        "conversation_id",
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_message",
        "brand_response",
        "resolution_type",
        "intent",
        "weak_intent",
        "timestamp",
    ]].to_dict("records")

    # Ensure customer_tweet_id is a plain string (JSON-serialisable)
    for m in metadata:
        m["customer_tweet_id"] = str(m["customer_tweet_id"])
        m["brand_tweet_id"] = str(m["brand_tweet_id"])

    store = FAISSStore()
    store.build(embeddings, metadata, intent_key="intent")
    store.save(index_path)

    t_elapsed = time.perf_counter() - t_start

    # ── Step 8: Save index manifest ───────────────────────────────────────────
    manifest = {
        "embedding_model": embed_model,
        "n_indexed": store.size,
        "n_excluded_golden": n_excluded,
        "vector_dim": store.dim,
        "index_type": "IndexFlatIP (exact cosine similarity)",
        "top_k_default": cfg["retrieval"]["top_k"],
        "index_path": str(index_path),
        "runtime_seconds": round(t_elapsed, 1),
        "seed": seed,
        "split_level": "tweet",
        "split_warning": "Conversation-level split not yet implemented",
    }
    manifest_path = processed_dir / "index_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("  Vectors indexed : %d", store.size)
    logger.info("  Excluded (golden): %d", n_excluded)
    logger.info("  Vector dim      : %d", store.dim)
    logger.info("  Index type      : IndexFlatIP (exact cosine)")
    logger.info("  Index saved to  : %s", index_path)
    logger.info("  Manifest saved  : %s", manifest_path)
    logger.info("  Runtime         : %.1f s", t_elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS retrieval index and train intent classifiers"
    )
    parser.add_argument(
        "--skip-classifier", action="store_true",
        help="Skip classifier training; load existing model (faster)"
    )
    args = parser.parse_args()
    main(args)
