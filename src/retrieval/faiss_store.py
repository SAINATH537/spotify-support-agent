"""
src/retrieval/faiss_store.py
─────────────────────────────────────────────────────────────────────────────
FAISS vector index for historical interaction retrieval.

Architecture:
  customer message → sentence-transformer → FAISS IndexFlatIP → top-K

Design decisions:
- IndexFlatIP (inner product) with normalised vectors == cosine similarity.
- Simplest correct index — no approximation, exact results, reproducible.
- Intent-aware retrieval: first filter by predicted intent, then retrieve.
  If intent-filtered results < intent_fallback_k, widen to full index.
- Metadata stored as a parallel list (JSON-serialisable).
- Evidence IDs and similarity scores returned for escalation policy.
- Golden test set IDs MUST be excluded before calling build().
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    FAISS inner-product index with intent-aware retrieval.

    Build:
        store = FAISSStore()
        store.build(embeddings, metadata_list)
        store.save("data/processed/faiss_index")

    Query:
        store = FAISSStore.load("data/processed/faiss_index")
        results = store.search(query_vec, top_k=5, intent="Playback / Audio")
    """

    def __init__(self) -> None:
        self._index = None          # faiss.IndexFlatIP
        self._metadata: list[dict] = []   # parallel to index rows
        self._intents: list[str] = []     # intent label per row (for filtering)
        self.dim: int = 0

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
        intent_key: str = "intent",
    ) -> None:
        """
        Build the FAISS index from pre-computed embeddings.

        Args:
            embeddings: float32 array (N, D), normalised
            metadata: list of dicts, one per embedding row
            intent_key: metadata key containing the intent label
        """
        import faiss

        assert embeddings.shape[0] == len(metadata), (
            f"Embedding rows ({embeddings.shape[0]}) != metadata rows ({len(metadata)})"
        )
        embeddings = embeddings.astype(np.float32)

        self.dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(embeddings)
        self._metadata = metadata
        self._intents = [m.get(intent_key, "") for m in metadata]
        logger.info(
            "Built FAISS index: %d vectors, dim=%d", self._index.ntotal, self.dim
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        intent: Optional[str] = None,
        fallback_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-K most similar interactions.

        If `intent` is provided, restrict to same-intent interactions first.
        If filtered results < fallback_k, widen to full index.

        Returns list of:
          {
            "interaction_id": str,
            "similarity": float,
            "customer_message": str,
            "brand_response": str,
            "resolution_type": str,
            "intent": str,
            ... other metadata fields
          }
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        qvec = query_vec.astype(np.float32).reshape(1, -1)

        if intent:
            results = self._search_with_intent(qvec, intent, top_k, fallback_k, min_similarity)
        else:
            results = self._search_full(qvec, top_k * 2)

        # Filter by minimum similarity
        results = [r for r in results if r["similarity"] >= min_similarity]
        return results[:top_k]

    def _search_full(self, qvec: np.ndarray, top_k: int) -> list[dict]:
        n = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(qvec, n)
        return [
            {**self._metadata[i], "similarity": float(s)}
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def _search_with_intent(
        self, qvec: np.ndarray, intent: str, top_k: int, fallback_k: int, min_sim: float
    ) -> list[dict]:
        import faiss
        # Find indices matching the intent
        intent_indices = [i for i, lab in enumerate(self._intents) if lab == intent]
        if not intent_indices:
            logger.debug("No entries for intent '%s', falling back to full search", intent)
            return self._search_full(qvec, top_k)

        # Build a temporary filtered index
        n_filtered = len(intent_indices)
        filtered_embs = np.zeros((n_filtered, self.dim), dtype=np.float32)
        temp_idx = faiss.IndexFlatIP(self.dim)
        for j, global_i in enumerate(intent_indices):
            vec = faiss.downcast_index(self._index).reconstruct(global_i)
            filtered_embs[j] = vec
        temp_idx.add(filtered_embs)

        k_query = min(top_k, n_filtered)
        scores, local_indices = temp_idx.search(qvec, k_query)
        filtered_results = [
            {**self._metadata[intent_indices[li]], "similarity": float(s)}
            for s, li in zip(scores[0], local_indices[0])
            if li >= 0
        ]

        # Fallback if too few good results
        good = [r for r in filtered_results if r["similarity"] >= min_sim]
        if len(good) < fallback_k:
            logger.debug(
                "Intent-filtered search returned %d results < fallback_k=%d. Widening.",
                len(good), fallback_k
            )
            full_results = self._search_full(qvec, top_k * 2)
            # Merge: intent-filtered first, then full, deduplicating
            seen = {r.get("interaction_id") for r in filtered_results}
            for r in full_results:
                if r.get("interaction_id") not in seen:
                    filtered_results.append(r)
                    seen.add(r.get("interaction_id"))

        return filtered_results

    def save(self, directory: str | Path) -> None:
        import faiss
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"metadata": self._metadata, "dim": self.dim}, f, ensure_ascii=False)
        logger.info("FAISSStore saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSStore":
        import faiss
        directory = Path(directory)
        obj = cls()
        obj._index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        obj._metadata = data["metadata"]
        obj.dim = data["dim"]
        obj._intents = [m.get("intent", "") for m in obj._metadata]
        logger.info(
            "FAISSStore loaded: %d vectors, dim=%d", obj._index.ntotal, obj.dim
        )
        return obj

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
