"""
src/data/load.py
─────────────────────────────────────────────────────────────────────────────
Chunked CSV loader for the TWCS dataset.

Design decisions:
- Process in chunks to avoid 516 MB CSV blowing RAM.
- Filter for SpotifyCares brand tweets as early as possible.
- Never load the full 2.8M-row frame unless explicitly requested.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}

DTYPES = {
    "tweet_id": "int64",
    "author_id": "str",
    "inbound": "bool",
    "created_at": "str",
    "text": "str",
    "response_tweet_id": "float64",   # NaN-capable
    "in_response_to_tweet_id": "float64",
}


def iter_chunks(
    csv_path: str | Path,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield raw chunks from the TWCS CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {csv_path}. "
            "Download twcs.csv and place it in data/raw/."
        )
    logger.info("Streaming %s in chunks of %d", csv_path, chunk_size)
    for chunk in pd.read_csv(
        csv_path,
        dtype=DTYPES,
        chunksize=chunk_size,
        low_memory=False,
    ):
        _validate_schema(chunk)
        yield chunk


def load_brand_tweets(
    csv_path: str | Path,
    brand: str = "SpotifyCares",
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Load only the tweets authored by `brand` from the full CSV."""
    parts: list[pd.DataFrame] = []
    total = 0
    for chunk in iter_chunks(csv_path, chunk_size):
        total += len(chunk)
        brand_rows = chunk[chunk["author_id"] == brand]
        if len(brand_rows):
            parts.append(brand_rows)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(DTYPES))
    logger.info(
        "Scanned %d rows — found %d brand tweets for '%s'", total, len(df), brand
    )
    return df


def load_inbound_for_brand(
    csv_path: str | Path,
    brand_tweet_ids: set[int],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Load all inbound (customer) tweets that are direct replies to the brand's tweets,
    OR that the brand replied to.
    These are identified via in_response_to_tweet_id matching a brand tweet id,
    or response_tweet_id matching a brand tweet id.
    """
    parts: list[pd.DataFrame] = []
    for chunk in iter_chunks(csv_path, chunk_size):
        inbound = chunk[chunk["inbound"] == True]
        # Customers who replied to brand
        replied_to_brand = inbound[
            inbound["in_response_to_tweet_id"].isin(brand_tweet_ids)
        ]
        # Customers whose tweet the brand responded to
        brand_responded = inbound[
            inbound["tweet_id"].isin(brand_tweet_ids)
        ]
        combined = pd.concat([replied_to_brand, brand_responded]).drop_duplicates("tweet_id")
        if len(combined):
            parts.append(combined)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(DTYPES))
    logger.info("Found %d customer tweets related to brand", len(df))
    return df


def load_tweets_by_ids(
    csv_path: str | Path,
    tweet_ids: set[int],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Load specific tweet_ids from the full CSV."""
    parts: list[pd.DataFrame] = []
    for chunk in iter_chunks(csv_path, chunk_size):
        hits = chunk[chunk["tweet_id"].isin(tweet_ids)]
        if len(hits):
            parts.append(hits)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(DTYPES))


def _validate_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
