#!/usr/bin/env python
"""
scripts/prepare_data.py
─────────────────────────────────────────────────────────────────────────────
Phase 2: End-to-end data preprocessing pipeline.

Steps:
  1. Load SpotifyCares brand tweets from raw CSV (chunked)
  2. Load matching customer (inbound) tweets
  3. Clean and normalise both sets
  4. Reconstruct customer→brand pairs
  5. Build conversation objects
  6. Build interaction records for retrieval
  7. Generate 250-example golden annotation template
  8. Save all processed files and statistics

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --sample 5000   # fast mode for reproduce
  python scripts/prepare_data.py --detect-lang   # slow: run langdetect
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.load import load_brand_tweets, load_inbound_for_brand
from src.data.clean import clean_dataframe
from src.data.conversations import (
    reconstruct_pairs,
    build_conversations,
    build_interactions,
    interactions_to_dataframe,
    conversations_to_jsonl,
)
from src.intent.taxonomy import IntentTaxonomy


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logging.getLogger("prepare_data")

    # ── Load config ──────────────────────────────────────────────────────────
    config_path = ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("random_seed", 42)
    random.seed(seed)

    raw_csv = ROOT / cfg["data"]["raw_csv"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    golden_dir = ROOT / cfg["data"]["golden_dir"]
    brand = cfg["data"]["brand"]
    chunk_size = cfg["data"]["chunk_size"]
    min_text_len = cfg["data"]["min_text_length"]
    ack_thresh = cfg["data"]["acknowledgement_threshold"]
    golden_n = cfg["data"]["golden_sample_size"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 2: Data Preprocessing")
    logger.info("Brand: %s  |  Seed: %d", brand, seed)
    logger.info("=" * 60)

    # ── Step 1: Load brand tweets ────────────────────────────────────────────
    logger.info("Step 1: Loading brand tweets from %s", raw_csv)
    brand_df = load_brand_tweets(raw_csv, brand=brand, chunk_size=chunk_size)
    logger.info("Brand tweets loaded: %d", len(brand_df))

    # ── Step 2: Load customer tweets ─────────────────────────────────────────
    logger.info("Step 2: Loading customer tweets...")
    # customer_tweet_ids = tweet IDs that SpotifyCares directly replied to
    # These come from brand tweets' in_response_to_tweet_id column (float64, clean).
    customer_tweet_ids: set[int] = set()
    for tid in brand_df["in_response_to_tweet_id"].dropna():
        customer_tweet_ids.add(int(tid))
    logger.info("Unique customer tweet IDs to fetch: %d", len(customer_tweet_ids))

    customer_df = load_inbound_for_brand(raw_csv, customer_tweet_ids, chunk_size=chunk_size)
    logger.info("Customer tweets loaded: %d", len(customer_df))

    # ── Step 3: Clean ────────────────────────────────────────────────────────
    logger.info("Step 3: Cleaning and normalising text...")
    brand_clean = clean_dataframe(
        brand_df,
        min_text_length=min_text_len,
        acknowledgement_threshold=ack_thresh,
        detect_lang=args.detect_lang,
    )
    customer_clean = clean_dataframe(
        customer_df,
        min_text_length=min_text_len,
        acknowledgement_threshold=ack_thresh,
        detect_lang=args.detect_lang,
    )

    # Remove duplicates (keep first occurrence)
    brand_clean = brand_clean[~brand_clean["is_duplicate"]].copy()
    customer_clean = customer_clean[~customer_clean["is_duplicate"]].copy()

    # Optional: English-only filter (log non-English but keep them for now)
    if args.detect_lang:
        n_non_en_cust = (~customer_clean["is_english"]).sum()
        logger.info("Non-English customer tweets flagged (not removed): %d", n_non_en_cust)

    # Optional sample for fast reproduce
    if args.sample and args.sample > 0:
        logger.warning(
            "SAMPLE MODE: restricting brand tweets to %d rows for fast reproduce.", args.sample
        )
        brand_clean = brand_clean.sample(n=min(args.sample, len(brand_clean)), random_state=seed)

    # ── Step 4: Reconstruct pairs ────────────────────────────────────────────
    logger.info("Step 4: Reconstructing customer→brand pairs...")
    pairs_df = reconstruct_pairs(brand_clean, customer_clean)

    # ── Step 5: Build conversations ──────────────────────────────────────────
    logger.info("Step 5: Building conversation objects...")
    conversations = build_conversations(pairs_df)

    # ── Step 6: Build retrieval interactions ─────────────────────────────────
    logger.info("Step 6: Building retrieval interaction records...")
    interactions = build_interactions(pairs_df)
    interactions_df = interactions_to_dataframe(interactions)

    # ── Step 7: Add weak intent labels (for annotation template only) ─────────
    logger.info("Step 7: Adding weak intent labels to interactions...")
    taxonomy = IntentTaxonomy(config_path)
    interactions_df["weak_intent"] = taxonomy.weak_label_batch(
        interactions_df["customer_message"].tolist()
    )
    # Also add to pairs_df
    pairs_df["weak_intent"] = taxonomy.weak_label_batch(
        pairs_df["customer_text_clean"].tolist()
    )

    # ── Step 8: Save processed files ─────────────────────────────────────────
    logger.info("Step 8: Saving processed files...")

    pairs_out = processed_dir / "spotify_customer_brand_pairs.csv"
    pairs_df.to_csv(pairs_out, index=False)
    logger.info("Saved pairs: %s (%d rows)", pairs_out, len(pairs_df))

    interactions_out = processed_dir / "spotify_interactions.csv"
    interactions_df.to_csv(interactions_out, index=False)
    logger.info("Saved interactions: %s (%d rows)", interactions_out, len(interactions_df))

    conversations_out = processed_dir / "spotify_two_turn_conversations.jsonl"
    conversations_to_jsonl(conversations, conversations_out)

    # ── Step 9: Generate golden annotation template ───────────────────────────
    logger.info("Step 9: Generating golden annotation template (%d examples)...", golden_n)
    _generate_golden_set(pairs_df, taxonomy, golden_dir, golden_n, seed)

    # ── Step 10: Save statistics ──────────────────────────────────────────────
    logger.info("Step 10: Saving statistics...")
    stats = {
        "brand": brand,
        "random_seed": seed,
        "brand_tweets": len(brand_clean),
        "customer_tweets": len(customer_clean),
        "reconstructed_pairs": len(pairs_df),
        "retrieval_worthy_pairs": int(pairs_df["retrieval_worthy"].sum()),
        "conversations": len(conversations),
        "interactions_for_retrieval": len(interactions_df),
        "resolution_type_counts": pairs_df["resolution_type"].value_counts().to_dict(),
        "weak_intent_counts": pairs_df["weak_intent"].value_counts().to_dict(),
        "golden_set_size": golden_n,
        "sample_mode": bool(args.sample),
        "sample_size": args.sample,
    }
    stats_path = processed_dir / "data_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Statistics saved to %s", stats_path)

    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info("Pairs: %d | Retrieval-worthy: %d | Conversations: %d",
                stats["reconstructed_pairs"],
                stats["retrieval_worthy_pairs"],
                stats["conversations"])
    logger.info("=" * 60)


def _generate_golden_set(
    pairs_df: pd.DataFrame,
    taxonomy: IntentTaxonomy,
    golden_dir: Path,
    n: int,
    seed: int,
) -> None:
    """
    Generate a stratified golden annotation template.
    Samples are stratified by weak_intent to ensure all classes are represented.
    Labels are WEAK (keyword heuristic) and must be manually verified.

    WARNING: Do NOT use these labels for training until manually verified.
    """
    import numpy as np

    # Filter: non-trivial, non-ack customer messages
    pool = pairs_df[
        (~pairs_df.get("is_ack", pd.Series(False, index=pairs_df.index))) &
        (pairs_df["customer_text_clean"].str.len() > 20)
    ].copy()

    # Stratified sample by weak_intent
    intents = pool["weak_intent"].unique()
    per_intent = max(1, n // len(intents))
    samples = []
    for intent in intents:
        subset = pool[pool["weak_intent"] == intent]
        k = min(per_intent, len(subset))
        samples.append(subset.sample(n=k, random_state=seed))
    sampled = pd.concat(samples, ignore_index=True)
    # Top up or trim to exactly n
    if len(sampled) < n:
        remaining = pool[~pool.index.isin(sampled.index)]
        extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)

    # Build annotation template
    records = []
    for i, row in sampled.iterrows():
        records.append({
            "example_id": f"ex_{i+1:04d}",
            "tweet_id": str(int(row["customer_tweet_id"])),
            "customer_message": row["customer_text_clean"],
            "historical_brand_reply": row["brand_text_clean"],
            "gold_intent": row["weak_intent"],         # PENDING HUMAN VERIFICATION
            "gold_intent_verified": False,              # must be set to True after review
            "should_escalate": "",                     # fill manually
            "escalation_reason": "",                   # fill manually
            "expected_action_or_facts": "",            # fill manually
            "difficulty": "",                          # easy / medium / hard
            "annotator_notes": "",                     # free text
            "sampling_stratum": row["weak_intent"],
            "resolution_type": row.get("resolution_type", ""),
        })

    golden_df = pd.DataFrame(records)
    out_path = golden_dir / "spotify_golden_set_annotation_template_250.csv"
    golden_df.to_csv(out_path, index=False)
    logging.getLogger("prepare_data").info(
        "Golden annotation template saved: %s (%d rows)\n"
        "  ⚠️  Labels are WEAK (keyword heuristic). Must be manually verified before evaluation.",
        out_path, len(golden_df)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Spotify support data")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N brand tweets for fast testing (default: all)")
    parser.add_argument("--detect-lang", action="store_true",
                        help="Run language detection (slow, requires langdetect)")
    args = parser.parse_args()
    main(args)
