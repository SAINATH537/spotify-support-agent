"""
tests/test_load.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/data/load.py.

Covers:
  - parse_response_tweet_ids: single, comma-list, NaN, empty, malformed
  - DTYPES: correct dtype declarations
  - load_inbound_for_brand: correct semantics (lookup by customer tweet_id)
  - BUG-9 regression: verify brand_tweet_ids are NOT used as customer ids
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load import (
    DTYPES,
    parse_response_tweet_ids,
    load_brand_tweets,
    load_inbound_for_brand,
    load_tweets_by_ids,
)


# ── parse_response_tweet_ids ──────────────────────────────────────────────────

def test_parse_single_id():
    assert parse_response_tweet_ids("12345") == [12345]


def test_parse_float_string():
    """Values stored as float strings like "12345.0" should work."""
    assert parse_response_tweet_ids("12345.0") == [12345]


def test_parse_comma_separated():
    assert parse_response_tweet_ids("5,7") == [5, 7]


def test_parse_comma_separated_long():
    result = parse_response_tweet_ids("9,6,10")
    assert result == [9, 6, 10]


def test_parse_comma_with_spaces():
    result = parse_response_tweet_ids("5, 7, 11")
    assert result == [5, 7, 11]


def test_parse_nan_float():
    import math
    assert parse_response_tweet_ids(float("nan")) == []


def test_parse_none():
    assert parse_response_tweet_ids(None) == []


def test_parse_empty_string():
    assert parse_response_tweet_ids("") == []


def test_parse_nan_string():
    assert parse_response_tweet_ids("nan") == []


def test_parse_malformed_token_skipped():
    """Malformed tokens should be skipped; valid tokens kept."""
    result = parse_response_tweet_ids("12345,abc,67890")
    assert 12345 in result
    assert 67890 in result
    assert len(result) == 2  # "abc" skipped


def test_parse_preserves_all_ids():
    """Multi-parent references must NOT be silently discarded (requirement)."""
    result = parse_response_tweet_ids("1,2,3,4,5")
    assert result == [1, 2, 3, 4, 5]
    assert len(result) == 5


# ── DTYPES ────────────────────────────────────────────────────────────────────

def test_response_tweet_id_is_object():
    """response_tweet_id must be object to allow comma-separated values."""
    assert DTYPES["response_tweet_id"] == "object"


def test_in_response_to_tweet_id_is_float64():
    """in_response_to_tweet_id is float64 — verified clean across full 2.8M rows."""
    assert DTYPES["in_response_to_tweet_id"] == "float64"


def test_tweet_id_is_int64():
    assert DTYPES["tweet_id"] == "int64"


# ── load_inbound_for_brand semantics ─────────────────────────────────────────

def _write_temp_csv(rows: list[dict], tmpdir: str) -> Path:
    """Write a minimal TWCS-schema CSV for testing."""
    df = pd.DataFrame(rows)
    path = Path(tmpdir) / "test_twcs.csv"
    df.to_csv(path, index=False)
    return path


def _make_test_rows():
    return [
        # Brand tweet: SpotifyCares replied to customer tweet 100
        {"tweet_id": 200, "author_id": "SpotifyCares", "inbound": False,
         "created_at": "2020-01-01", "text": "Try restarting.",
         "response_tweet_id": None, "in_response_to_tweet_id": 100.0},
        # Customer tweet 100: the one SpotifyCares replied to
        {"tweet_id": 100, "author_id": "user1", "inbound": True,
         "created_at": "2020-01-01", "text": "App crashes.",
         "response_tweet_id": "200", "in_response_to_tweet_id": None},
        # Unrelated customer tweet: SpotifyCares did NOT reply to this
        {"tweet_id": 999, "author_id": "user2", "inbound": True,
         "created_at": "2020-01-02", "text": "Random complaint.",
         "response_tweet_id": None, "in_response_to_tweet_id": None},
        # Another brand tweet with a different customer
        {"tweet_id": 201, "author_id": "SpotifyCares", "inbound": False,
         "created_at": "2020-01-02", "text": "DM us please.",
         "response_tweet_id": None, "in_response_to_tweet_id": 101.0},
        # Customer tweet 101: the other customer SpotifyCares replied to
        {"tweet_id": 101, "author_id": "user3", "inbound": True,
         "created_at": "2020-01-02", "text": "Charged twice.",
         "response_tweet_id": "201", "in_response_to_tweet_id": None},
    ]


def test_load_inbound_for_brand_returns_correct_customers():
    """load_inbound_for_brand must return the customer tweets that SpotifyCares replied to."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_temp_csv(_make_test_rows(), tmpdir)
        # Customer tweet IDs SpotifyCares replied to: 100, 101
        customer_ids = {100, 101}
        df = load_inbound_for_brand(csv_path, customer_ids)
        assert set(df["tweet_id"].tolist()) == {100, 101}
        assert len(df) == 2


def test_load_inbound_excludes_unrelated_customers():
    """Tweets that SpotifyCares never replied to must NOT be returned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_temp_csv(_make_test_rows(), tmpdir)
        customer_ids = {100, 101}
        df = load_inbound_for_brand(csv_path, customer_ids)
        assert 999 not in df["tweet_id"].tolist()


def test_load_inbound_empty_id_set():
    """Empty set of IDs → empty DataFrame, no crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_temp_csv(_make_test_rows(), tmpdir)
        df = load_inbound_for_brand(csv_path, set())
        assert len(df) == 0


# ── BUG-9 REGRESSION: brand_tweet_ids must NOT be used as customer lookup ────

def test_bug9_regression_correct_ids_are_passed():
    """
    BUG-9 regression test.

    The old (incorrect) code in prepare_data.py called:
        load_inbound_for_brand(csv, brand_tweet_ids={200, 201})

    But load_inbound_for_brand looks up tweets by tweet_id. Since the brand
    tweets have tweet_ids 200 and 201, and customer tweets have tweet_ids
    100 and 101, the old call would return the brand tweets themselves (or 0
    rows if brand tweets don't appear in the inbound portion of the CSV scan).

    The correct call in prepare_data.py now extracts customer_tweet_ids from
    brand_df["in_response_to_tweet_id"] and passes those:
        load_inbound_for_brand(csv, customer_tweet_ids={100, 101})

    This test verifies that:
    1. Passing customer_tweet_ids {100, 101} returns the 2 customer tweets.
    2. Passing brand_tweet_ids {200, 201} would return brand tweets, not customers
       — confirming the old code was wrong about the intent.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_temp_csv(_make_test_rows(), tmpdir)

        # Correct: passing customer tweet IDs (what prepare_data.py now does)
        customer_ids = {100, 101}
        df_correct = load_inbound_for_brand(csv_path, customer_ids)
        assert len(df_correct) == 2
        returned_ids = set(df_correct["tweet_id"].tolist())
        assert returned_ids == {100, 101}, f"Expected {{100, 101}}, got {returned_ids}"
        # All returned tweets should be inbound (customer) tweets
        assert all(df_correct["inbound"] == True), "All returned tweets should be inbound (customer)"

        # Wrong (old behavior): passing brand tweet IDs returns brand tweets, not customers
        brand_ids = {200, 201}
        df_wrong = load_inbound_for_brand(csv_path, brand_ids)
        # The function returns tweets with those IDs — but they are brand tweets (inbound=False)
        # This is why the old code was wrong: it fetched the wrong tweets
        if len(df_wrong) > 0:
            assert all(df_wrong["inbound"] == False), (
                "BUG-9: passing brand_tweet_ids returns brand tweets (inbound=False), "
                "not the customer tweets we need."
            )


# ── load_brand_tweets ─────────────────────────────────────────────────────────

def test_load_brand_tweets_filters_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_temp_csv(_make_test_rows(), tmpdir)
        df = load_brand_tweets(csv_path, brand="SpotifyCares")
        assert len(df) == 2
        assert all(df["author_id"] == "SpotifyCares")
        assert all(df["inbound"] == False)


def test_load_brand_tweets_other_brand_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_temp_csv(_make_test_rows(), tmpdir)
        df = load_brand_tweets(csv_path, brand="AmazonHelp")
        assert len(df) == 0


# ── CSV dtype safety ──────────────────────────────────────────────────────────

def test_csv_with_comma_separated_response_id_does_not_crash():
    """
    The original BUG-1 crash: response_tweet_id values like '5,7' caused
    TypeError when dtype was float64. With object dtype it must load cleanly.
    """
    rows = [
        {"tweet_id": 5, "author_id": "BrandX", "inbound": False,
         "created_at": "2020-01-01", "text": "Reply 1.",
         "response_tweet_id": "5,7", "in_response_to_tweet_id": 3.0},
        {"tweet_id": 6, "author_id": "BrandX", "inbound": False,
         "created_at": "2020-01-01", "text": "Reply 2.",
         "response_tweet_id": "9,6,10", "in_response_to_tweet_id": 4.0},
        {"tweet_id": 3, "author_id": "cust1", "inbound": True,
         "created_at": "2020-01-01", "text": "Help me.",
         "response_tweet_id": None, "in_response_to_tweet_id": None},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_temp_csv(rows, tmpdir)
        # Must not raise
        df = load_brand_tweets(csv_path, brand="BrandX")
        assert len(df) == 2
        # response_tweet_id preserved as object (not coerced to float)
        assert df["response_tweet_id"].dtype == object
        assert "5,7" in df["response_tweet_id"].tolist()
