"""
src/retrieval/embeddings.py
─────────────────────────────────────────────────────────────────────────────
Batch embedding utilities for building the FAISS index.

Design decisions:
- Embeddings are stored as float32 numpy arrays on disk.
- The golden test set tweet IDs are excluded from the index
  to prevent data leakage (cannot evaluate on indexed examples).
- Deterministic: no randomness; same input → same output always.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    cache_dir: str = "data/processed/embedding_cache",
    cache_key_prefix: str = "",
) -> np.ndarray:
    """
    Embed a list of texts using a sentence-transformer model.
    Results are cached to disk to avoid re-computation.

    Returns: float32 numpy array of shape (N, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    key_str = cache_key_prefix + "||".join(texts) + model_name
    cache_key = hashlib.md5(key_str.encode()).hexdigest()[:16]
    cache_path = cache_dir_path / f"{cache_key}.npy"

    if cache_path.exists():
        logger.info("Cache hit: loading embeddings from %s", cache_path)
        return np.load(cache_path).astype(np.float32)

    logger.info("Embedding %d texts with %s ...", len(texts), model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(cache_path, embeddings)
    logger.info("Saved embeddings to %s  shape=%s", cache_path, embeddings.shape)
    return embeddings


def embed_single(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    _model_cache: dict = {},
) -> np.ndarray:
    """
    Embed a single query text at inference time.
    Model is loaded once and reused (simple in-process cache via mutable default).
    """
    from sentence_transformers import SentenceTransformer
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    vec = _model_cache[model_name].encode(
        [text], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    return vec[0]


def exclude_ids(
    texts: list[str],
    ids: list[str],
    exclude_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Filter out items whose id is in exclude_ids."""
    filtered_texts, filtered_ids = zip(
        *[(t, i) for t, i in zip(texts, ids) if i not in exclude_ids]
    ) if texts else ([], [])
    n_excluded = len(texts) - len(filtered_texts)
    if n_excluded:
        logger.info("Excluded %d items from embedding (leakage prevention)", n_excluded)
    return list(filtered_texts), list(filtered_ids)
