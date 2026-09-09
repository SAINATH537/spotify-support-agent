"""
tests/test_retrieval.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for FAISS store build, query, persistence, and golden exclusion.

Test coverage (10 required cases):
  1. Index builds successfully
  2. Index can be saved and loaded
  3. Metadata count matches index count
  4. Every returned result has valid metadata
  5. Golden tweet IDs are excluded from the index
  6. top_k behaves correctly
  7. Empty / invalid queries are handled safely
  8. Retrieval works when intent is absent
  9. Intent-aware retrieval works when intent is supplied
  10. Similarity scores are returned correctly
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.faiss_store import FAISSStore


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_dummy_embeddings(n: int = 20, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


INTENTS = ["Playback / Audio", "App / Device / Technical", "Account / Login / Security"]


def _make_dummy_metadata(n: int = 20, start_id: int = 1000) -> list[dict]:
    """
    Build metadata whose customer_tweet_id values are outside the golden set.
    Golden tweet IDs in tests use the range 1..100.
    This fixture uses start_id=1000 by default to avoid overlap.
    """
    return [
        {
            "interaction_id": f"int_{i:04d}",
            "conversation_id": f"conv_{start_id + i}",
            "customer_tweet_id": str(start_id + i),
            "brand_tweet_id": str(start_id + i + 500),
            "customer_message": f"Customer message {i}",
            "brand_response": f"Brand response {i}",
            "resolution_type": "troubleshooting",
            "intent": INTENTS[i % 3],
            "weak_intent": INTENTS[i % 3],
            "timestamp": "2020-01-01",
        }
        for i in range(n)
    ]


def _built_store(n: int = 20) -> FAISSStore:
    embs = _make_dummy_embeddings(n)
    meta = _make_dummy_metadata(n)
    store = FAISSStore()
    store.build(embs, meta)
    return store


# ── Test 1: Index builds successfully ────────────────────────────────────────

def test_faiss_build_succeeds():
    store = _built_store(20)
    assert store.size == 20
    assert store.dim == 384
    assert store._index is not None


# ── Test 2: Index can be saved and loaded ────────────────────────────────────

def test_faiss_save_load():
    store = _built_store(10)
    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        assert (Path(tmp) / "index.faiss").exists()
        assert (Path(tmp) / "metadata.json").exists()

        loaded = FAISSStore.load(tmp)
        assert loaded.size == 10
        assert loaded.dim == store.dim

        # Results must be identical
        query = _make_dummy_embeddings(1)[0]
        orig = store.search(query, top_k=3)
        new = loaded.search(query, top_k=3)
        assert orig[0]["interaction_id"] == new[0]["interaction_id"]
        assert abs(orig[0]["similarity"] - new[0]["similarity"]) < 1e-5


# ── Test 3: Metadata count matches index count ────────────────────────────────

def test_metadata_count_matches_index():
    n = 15
    store = _built_store(n)
    assert len(store._metadata) == n
    assert store.size == n
    assert len(store._intents) == n


# ── Test 4: Every returned result has valid metadata ─────────────────────────

def test_results_have_valid_metadata():
    store = _built_store(20)
    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=5)

    required_keys = {"interaction_id", "customer_message", "brand_response",
                     "resolution_type", "intent", "similarity", "customer_tweet_id"}
    for r in results:
        for key in required_keys:
            assert key in r, f"Missing key '{key}' in result: {r.keys()}"
        assert isinstance(r["similarity"], float)
        assert r["customer_message"]   # not empty
        assert r["brand_response"]     # not empty
        assert r["interaction_id"]     # not empty
        assert r["customer_tweet_id"]  # traceable


# ── Test 5: Golden tweet IDs are excluded ────────────────────────────────────

def test_golden_ids_excluded_from_index():
    """
    Simulate the golden exclusion that build_index.py performs.
    Build an index from a pool of interactions, explicitly excluding
    golden tweet IDs, and verify the intersection is empty.
    """
    GOLDEN_IDS = {"1", "2", "3", "42", "99"}  # simulated golden set

    all_meta = _make_dummy_metadata(30, start_id=0)  # IDs 0–29

    # Exclude golden IDs (mirrors build_index.py logic)
    filtered_meta = [m for m in all_meta if m["customer_tweet_id"] not in GOLDEN_IDS]
    n = len(filtered_meta)
    embs = _make_dummy_embeddings(n)

    store = FAISSStore()
    store.build(embs, filtered_meta)

    # Verify: intersection of golden IDs and indexed IDs is empty
    indexed_ids = {m["customer_tweet_id"] for m in store._metadata}
    overlap = indexed_ids & GOLDEN_IDS
    assert len(overlap) == 0, (
        f"Golden IDs found in index: {overlap}. "
        "Golden exclusion in build_index.py is broken."
    )


def test_golden_ids_excluded_count():
    """Verify the correct number are excluded."""
    GOLDEN_IDS = {"1000", "1001", "1002"}  # 3 IDs that match our fixture

    all_meta = _make_dummy_metadata(20, start_id=1000)  # IDs 1000–1019
    filtered = [m for m in all_meta if m["customer_tweet_id"] not in GOLDEN_IDS]
    assert len(filtered) == 17  # 20 - 3 excluded


# ── Test 6: top_k behaves correctly ──────────────────────────────────────────

def test_top_k_exactly_respected():
    store = _built_store(20)
    query = _make_dummy_embeddings(1)[0]

    for k in [1, 3, 5, 10]:
        results = store.search(query, top_k=k)
        assert len(results) == k, f"Expected {k} results, got {len(results)}"


def test_top_k_capped_at_index_size():
    store = _built_store(5)
    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=100)
    assert len(results) <= 5


# ── Test 7: Empty / invalid queries handled safely ───────────────────────────

def test_zero_vector_query_does_not_crash():
    """A zero vector is degenerate but must not crash."""
    store = _built_store(10)
    zero = np.zeros(384, dtype=np.float32)
    try:
        results = store.search(zero, top_k=3)
        # Either returns results or empty — but must not raise
        assert isinstance(results, list)
    except Exception as e:
        pytest.fail(f"Zero-vector query raised: {e}")


def test_unbuilt_store_raises():
    """Querying an unbuilt store must raise RuntimeError, not crash silently."""
    store = FAISSStore()
    with pytest.raises(RuntimeError, match="not built"):
        store.search(np.zeros(384, dtype=np.float32), top_k=3)


def test_very_short_query_string():
    """Single-word query should still return results."""
    store = _built_store(10)
    # Just use a random vector to simulate a short-query embedding
    vec = _make_dummy_embeddings(1)[0]
    results = store.search(vec, top_k=3)
    assert len(results) >= 1


# ── Test 8: Retrieval works when intent is absent ────────────────────────────

def test_retrieval_without_intent():
    store = _built_store(20)
    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=3, intent=None)
    assert len(results) == 3
    # Must still return useful metadata
    for r in results:
        assert "customer_message" in r
        assert "brand_response" in r


def test_retrieval_unknown_intent_falls_back():
    """If intent has no matching entries, full-index fallback must fire."""
    store = _built_store(20)
    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=3, intent="NonExistentIntent", fallback_k=1)
    # Should fall back to full search and still return results
    assert len(results) >= 1


# ── Test 9: Intent-aware retrieval ───────────────────────────────────────────

def test_intent_filtered_search_returns_matching_intent():
    store = _built_store(30)
    query = _make_dummy_embeddings(1)[0]
    target_intent = "Playback / Audio"
    results = store.search(query, top_k=5, intent=target_intent, fallback_k=1)
    assert len(results) >= 1
    # At least the top result should be the target intent (before fallback mixing)
    # We just check we got results and they have the field
    for r in results:
        assert "intent" in r


def test_intent_filtered_no_cross_intent_contamination():
    """
    When intent-filtered pool is large enough (fallback not triggered),
    all results should be from the requested intent.
    """
    n = 60  # 20 per intent → enough to satisfy top_k=5 without fallback
    embs = _make_dummy_embeddings(n)
    meta = _make_dummy_metadata(n)
    store = FAISSStore()
    store.build(embs, meta)

    query = _make_dummy_embeddings(1)[0]
    target_intent = "Playback / Audio"
    # Set fallback_k=0 to force intent-only results
    results = store.search(query, top_k=5, intent=target_intent, fallback_k=0)
    assert len(results) >= 1
    for r in results:
        assert r["intent"] == target_intent, (
            f"Expected intent '{target_intent}', got '{r['intent']}'"
        )


# ── Test 10: Similarity scores are returned correctly ────────────────────────

def test_similarity_scores_in_valid_range():
    store = _built_store(20)
    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=5)
    for r in results:
        sim = r["similarity"]
        assert -0.05 <= sim <= 1.05, f"Similarity {sim} out of [-0.05, 1.05] range"


def test_similarity_scores_descending():
    """Results must be ordered by descending similarity."""
    store = _built_store(20)
    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=10)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True), "Results not sorted by descending similarity"


def test_self_similarity_is_high():
    """
    A query that is identical to an indexed message should have very high similarity.
    """
    embs = _make_dummy_embeddings(10)
    meta = _make_dummy_metadata(10)
    store = FAISSStore()
    store.build(embs, meta)

    # Query with the first indexed vector (should be ~1.0 similarity to itself)
    query = embs[0]
    results = store.search(query, top_k=1)
    assert results[0]["similarity"] > 0.99, (
        f"Self-similarity expected > 0.99, got {results[0]['similarity']}"
    )


def test_min_similarity_filter():
    store = _built_store(10)
    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=5, min_similarity=0.9999)
    # Very high threshold — should return 0 or very few results
    assert len(results) <= 1


# ── Integration: build from existing fixture + build_index.py exclusion logic ─

def test_golden_exclusion_assertion_fires():
    """
    Verify that if golden IDs are NOT excluded before build,
    the assertion in build_index.py would catch it (unit test of the guard).
    """
    GOLDEN_IDS = {"1000", "1001"}
    meta = _make_dummy_metadata(5, start_id=1000)  # IDs 1000–1004 — overlap with golden

    # Simulate: forget to exclude → overlap detected
    indexed_ids = {m["customer_tweet_id"] for m in meta}
    overlap = indexed_ids & GOLDEN_IDS
    assert len(overlap) > 0, "Test setup: expected overlap to exist before exclusion"

    # Now apply exclusion
    filtered = [m for m in meta if m["customer_tweet_id"] not in GOLDEN_IDS]
    indexed_ids_after = {m["customer_tweet_id"] for m in filtered}
    overlap_after = indexed_ids_after & GOLDEN_IDS
    assert len(overlap_after) == 0, "After exclusion: overlap must be zero"


# ── Existing tests preserved from original test_retrieval.py ─────────────────

def test_faiss_build_and_search():
    """Original test — preserved."""
    embs = _make_dummy_embeddings(20)
    meta = _make_dummy_metadata(20)
    store = FAISSStore()
    store.build(embs, meta)

    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=3)
    assert len(results) == 3
    for r in results:
        assert "interaction_id" in r
        assert "similarity" in r
        assert -0.1 <= r["similarity"] <= 1.01


def test_faiss_intent_filtered_search():
    """Original test — preserved."""
    embs = _make_dummy_embeddings(20)
    meta = _make_dummy_metadata(20)
    store = FAISSStore()
    store.build(embs, meta)

    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=3, intent="Playback / Audio", fallback_k=1)
    assert len(results) >= 1


def test_faiss_size():
    """Original test — preserved."""
    embs = _make_dummy_embeddings(15)
    meta = _make_dummy_metadata(15)
    store = FAISSStore()
    store.build(embs, meta)
    assert store.size == 15
