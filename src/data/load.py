"""
src/data/load.py
─────────────────────────────────────────────────────────────────────────────
Chunked CSV loader for the TWCS dataset.

Design decisions:
- Process in chunks to avoid 516 MB CSV blowing RAM.
- Filter for SpotifyCares brand tweets as early as possible.
- Never load the full 2.8M-row frame unless explicitly requested.

TWCS column semantics (verified against full 2.8M-row scan):
  tweet_id                  — unique tweet ID (int64, always clean)
  author_id                 — brand or customer username (str)
  inbound                   — True = customer tweet, False = brand tweet
  created_at                — timestamp string
  text                      — tweet text
  response_tweet_id         — ID(s) of tweet(s) this was answered by.
                              COMMA-SEPARATED when a tweet got multiple
                              replies (e.g. "5,7" or "9,6,10"). ~18k rows
                              per 200k chunk have this format.
                              → loaded as object dtype; do NOT cast to float.
  in_response_to_tweet_id   — single parent tweet ID this tweet replied to.
                              All values are cleanly float-castable or NaN
                              (verified by full-dataset scan). Loaded as
                              float64 so NaN handling works natively.

Data relationships:
  Brand tweet:   inbound=False,  in_response_to_tweet_id = customer tweet id
  Customer tweet: inbound=True,  in_response_to_tweet_id = prior brand tweet (if follow-up)
                                  response_tweet_id       = brand reply tweet id(s)

Finding customer tweets for a brand:
  brand_df["in_response_to_tweet_id"] → the customer tweet IDs the brand replied to.
  These customer tweet IDs are what we need to load from the CSV.
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

# response_tweet_id is object because it contains comma-separated multi-values.
# in_response_to_tweet_id is float64 — all values verified clean (single ID or NaN).
DTYPES = {
    "tweet_id": "int64",
    "author_id": "str",
    "inbound": "bool",
    "created_at": "str",
    "text": "str",
    "response_tweet_id": "object",       # comma-separated lists possible: "5,7", "9,6,10"
    "in_response_to_tweet_id": "float64", # single numeric ID or NaN — safe to cast
}


def parse_response_tweet_ids(val) -> list[int]:
    """
    Parse a response_tweet_id value into a list of integer tweet IDs.

    Handles:
      - NaN / None / empty string → []
      - Single integer: "12345" → [12345]
      - Comma-separated list: "5,7" → [5, 7], "9,6,10" → [9, 6, 10]
      - Malformed tokens (non-numeric) → skipped, logged at DEBUG

    This preserves ALL multi-parent references — none are silently discarded.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    result = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(float(token)))
        except (ValueError, TypeError):
            logger.debug("Skipping malformed response_tweet_id token: %r", token)
    return result


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
    customer_tweet_ids: set[int],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Load customer (inbound) tweets by their tweet_id.

    TWCS data semantics — how to find customer tweets for a brand:
      1. Load all brand tweets (author_id == "SpotifyCares", inbound=False).
      2. Each brand tweet has in_response_to_tweet_id = the customer tweet it answered.
      3. Collect those customer tweet IDs from the brand DataFrame (caller's job).
      4. This function loads those specific inbound tweets by tweet_id.

    Why NOT filter by inbound[in_response_to_tweet_id].isin(brand_tweet_ids):
      That would find customer tweets that replied TO a brand tweet (follow-up
      messages in a thread), not the initial customer tweets that triggered a
      brand response. Those are a different set and would cause incorrect pairing.

    Args:
        csv_path: path to twcs.csv
        customer_tweet_ids: set of tweet_ids from brand_df["in_response_to_tweet_id"]
                            (the customer tweet IDs the brand actually replied to)
        chunk_size: rows per chunk
    """
    if not customer_tweet_ids:
        logger.warning("load_inbound_for_brand: called with empty customer_tweet_ids set")
        return pd.DataFrame(columns=list(DTYPES))

    parts: list[pd.DataFrame] = []
    for chunk in iter_chunks(csv_path, chunk_size):
        hits = chunk[chunk["tweet_id"].isin(customer_tweet_ids)]
        if len(hits):
            parts.append(hits)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(DTYPES))
    logger.info(
        "Loaded %d customer tweets (of %d requested ids)", len(df), len(customer_tweet_ids)
    )
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
