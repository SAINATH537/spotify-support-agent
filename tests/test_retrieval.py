"""
tests/test_retrieval.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for FAISS store build and query.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.faiss_store import FAISSStore


def _make_dummy_embeddings(n: int = 20, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim)).astype(np.float32)
    # Normalise (cosine similarity)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_dummy_metadata(n: int = 20) -> list[dict]:
    intents = ["Playback / Audio", "App / Device / Technical", "Account / Login / Security"]
    return [
        {
            "interaction_id": f"int_{i:04d}",
            "customer_message": f"Customer message {i}",
            "brand_response": f"Brand response {i}",
            "resolution_type": "troubleshooting",
            "intent": intents[i % 3],
        }
        for i in range(n)
    ]


def test_faiss_build_and_search():
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
        assert -0.1 <= r["similarity"] <= 1.01  # cosine similarity range


def test_faiss_intent_filtered_search():
    embs = _make_dummy_embeddings(20)
    meta = _make_dummy_metadata(20)
    store = FAISSStore()
    store.build(embs, meta)

    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=3, intent="Playback / Audio", fallback_k=1)
    # All results should be Playback / Audio OR fallback (full search)
    # Just check we get results back
    assert len(results) >= 1


def test_faiss_save_load():
    embs = _make_dummy_embeddings(10)
    meta = _make_dummy_metadata(10)
    store = FAISSStore()
    store.build(embs, meta)

    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        loaded = FAISSStore.load(tmp)
        assert loaded.size == 10

        query = _make_dummy_embeddings(1)[0]
        results_orig = store.search(query, top_k=3)
        results_loaded = loaded.search(query, top_k=3)
        # Same top result
        assert results_orig[0]["interaction_id"] == results_loaded[0]["interaction_id"]


def test_faiss_min_similarity_filter():
    embs = _make_dummy_embeddings(10)
    meta = _make_dummy_metadata(10)
    store = FAISSStore()
    store.build(embs, meta)

    query = _make_dummy_embeddings(1)[0]
    results = store.search(query, top_k=5, min_similarity=0.99)
    # Very high threshold — should return 0 or 1 results
    assert len(results) <= 1


def test_faiss_size():
    embs = _make_dummy_embeddings(15)
    meta = _make_dummy_metadata(15)
    store = FAISSStore()
    store.build(embs, meta)
    assert store.size == 15
