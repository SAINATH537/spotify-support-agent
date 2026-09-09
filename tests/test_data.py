"""
tests/test_data.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for data loading, cleaning, and conversation reconstruction.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.clean import normalise_text, is_acknowledgement, clean_dataframe
from src.data.conversations import (
    classify_resolution, is_retrieval_worthy,
    reconstruct_pairs, build_conversations, build_interactions,
)


# ── normalise_text ────────────────────────────────────────────────────────────

def test_url_replacement():
    text = "Check this link https://open.spotify.com/track/123 for details"
    result = normalise_text(text, replace_urls=True)
    assert "<URL>" in result
    assert "https://" not in result


def test_mention_replacement():
    text = "@SpotifyCares I need help with @friend's playlist"
    result = normalise_text(text, replace_mentions=True)
    assert "@SpotifyCares" not in result
    assert "<USER>" in result


def test_emoji_preserved():
    text = "My app keeps crashing 😭 please help!"
    result = normalise_text(text)
    assert "😭" in result


def test_empty_text():
    assert normalise_text("") == ""
    assert normalise_text(None) == ""


def test_whitespace_collapsed():
    text = "hello    world\t\there"
    result = normalise_text(text)
    assert "  " not in result


# ── is_acknowledgement ────────────────────────────────────────────────────────

def test_ack_short():
    assert is_acknowledgement("ok", threshold=15)
    assert is_acknowledgement("thanks!", threshold=15)


def test_ack_phrase():
    assert is_acknowledgement("you're welcome", threshold=15)
    assert is_acknowledgement("no problem", threshold=15)


def test_not_ack():
    assert not is_acknowledgement("I can't log into my Spotify account", threshold=15)


# ── clean_dataframe ───────────────────────────────────────────────────────────

def _make_df(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "tweet_id": list(range(len(texts))),
        "author_id": ["test"] * len(texts),
        "inbound": [True] * len(texts),
        "created_at": ["2020-01-01"] * len(texts),
        "text": texts,
        "response_tweet_id": [None] * len(texts),
        "in_response_to_tweet_id": [None] * len(texts),
    })


def test_clean_dataframe_columns():
    df = _make_df(["Hello @User check https://spotify.com", "thanks"])
    out = clean_dataframe(df)
    assert "text_clean" in out.columns
    assert "is_ack" in out.columns
    assert "too_short" in out.columns
    assert "is_duplicate" in out.columns


def test_clean_dataframe_no_silent_delete():
    """Rows should be flagged, never silently dropped."""
    df = _make_df(["ok", "x", "normal message here"])
    out = clean_dataframe(df)
    assert len(out) == 3  # all rows still present


def test_clean_dataframe_duplicate_flagging():
    df = _make_df(["Hello world", "Hello world"])
    df["tweet_id"] = [1, 1]  # same tweet_id
    out = clean_dataframe(df)
    assert out["is_duplicate"].sum() == 1


# ── classify_resolution ───────────────────────────────────────────────────────

def test_resolution_troubleshooting():
    text = "Try restarting the app and logging out then back in."
    assert classify_resolution(text) == "troubleshooting"


def test_resolution_private_escalation():
    text = "Please DM us so we can take a closer look at your account."
    assert classify_resolution(text) == "private_escalation"


def test_resolution_acknowledgement():
    assert classify_resolution("Thanks!") == "acknowledgement"
    assert classify_resolution("ok") == "acknowledgement"


def test_resolution_informational():
    text = "You can find your listening history at spotify.com/account."
    assert classify_resolution(text) == "informational"


def test_resolution_clarification():
    text = "Can you tell us which device you're using?"
    assert classify_resolution(text) == "clarification_request"


# ── is_retrieval_worthy ───────────────────────────────────────────────────────

def test_retrieval_worthy():
    assert is_retrieval_worthy("troubleshooting")
    assert is_retrieval_worthy("informational")
    assert is_retrieval_worthy("clarification_request")
    assert not is_retrieval_worthy("acknowledgement")
    assert not is_retrieval_worthy("closing")
    assert not is_retrieval_worthy("private_escalation")


# ── reconstruct_pairs ────────────────────────────────────────────────────────

def _make_pairs_fixture():
    brand_df = pd.DataFrame({
        "tweet_id": [200, 201],
        "author_id": ["SpotifyCares", "SpotifyCares"],
        "inbound": [False, False],
        "created_at": ["2020-01-01", "2020-01-02"],
        "text": [
            "Try restarting the Spotify app and clearing the cache.",
            "Thanks for reaching out! Please DM us.",
        ],
        "text_clean": [
            "Try restarting the Spotify app and clearing the cache.",
            "Thanks for reaching out! Please DM us.",
        ],
        "response_tweet_id": [None, None],
        "in_response_to_tweet_id": [100, 101],
    })
    customer_df = pd.DataFrame({
        "tweet_id": [100, 101],
        "author_id": ["user1", "user2"],
        "inbound": [True, True],
        "created_at": ["2020-01-01", "2020-01-02"],
        "text": ["My app keeps crashing on Android", "I got charged twice this month"],
        "text_clean": ["My app keeps crashing on Android", "I got charged twice this month"],
        "response_tweet_id": [200, 201],
        "in_response_to_tweet_id": [None, None],
    })
    return brand_df, customer_df


def test_reconstruct_pairs_basic():
    brand_df, customer_df = _make_pairs_fixture()
    pairs = reconstruct_pairs(brand_df, customer_df)
    assert len(pairs) == 2
    assert "customer_tweet_id" in pairs.columns
    assert "brand_tweet_id" in pairs.columns
    assert "resolution_type" in pairs.columns
    assert "retrieval_worthy" in pairs.columns


def test_reconstruct_pairs_resolution_types():
    brand_df, customer_df = _make_pairs_fixture()
    pairs = reconstruct_pairs(brand_df, customer_df)
    types = pairs["resolution_type"].tolist()
    assert "troubleshooting" in types
    assert "private_escalation" in types


def test_build_conversations():
    brand_df, customer_df = _make_pairs_fixture()
    pairs = reconstruct_pairs(brand_df, customer_df)
    convs = build_conversations(pairs)
    assert len(convs) == 2
    for conv in convs:
        assert len(conv.messages) == 2
        speakers = [m.speaker for m in conv.messages]
        assert "customer" in speakers
        assert "brand" in speakers


def test_build_interactions_filters_trivial():
    """Only retrieval-worthy interactions should be returned."""
    brand_df, customer_df = _make_pairs_fixture()
    pairs = reconstruct_pairs(brand_df, customer_df)
    interactions = build_interactions(pairs)
    # Only troubleshooting/informational/clarification are retrieval-worthy
    for inter in interactions:
        assert inter.retrieval_worthy


def test_conversations_to_jsonl():
    brand_df, customer_df = _make_pairs_fixture()
    pairs = reconstruct_pairs(brand_df, customer_df)
    convs = build_conversations(pairs)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_convs.jsonl"
        from src.data.conversations import conversations_to_jsonl
        conversations_to_jsonl(convs, str(path))
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert "conversation_id" in data
        assert "messages" in data
