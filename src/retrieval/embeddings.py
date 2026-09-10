"""
src/retrieval/embeddings.py
─────────────────────────────────────────────────────────────────────────────
Batch and single-query embedding utilities.

Design decisions:
- Embeddings are stored as float32 numpy arrays on disk.
- The golden test set tweet IDs are excluded from the index
  to prevent data leakage (cannot evaluate on indexed examples).
- Deterministic: no randomness; same input → same output always.
- On Windows, torch/sentence_transformers may fail to load if sklearn/joblib
  has already been imported in-process (c10.dll init order conflict).
  embed_single falls back to a subprocess-based embedding to avoid this.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
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
    import torch  # must precede sentence_transformers on Windows (DLL init order)
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


# In-process model cache — populated if torch loads cleanly
_model_cache: dict = {}


def embed_single(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Embed a single query text at inference time.

    Tries in-process first (fast, cached model). If torch fails to load
    (Windows c10.dll init order conflict after sklearn), falls back to
    a subprocess that imports torch before any sklearn code.
    """
    global _model_cache

    # Try in-process (works if torch was imported before sklearn)
    if model_name in _model_cache:
        vec = _model_cache[model_name].encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        return vec[0]

    try:
        import torch  # must be first
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
        vec = _model_cache[model_name].encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        return vec[0]
    except OSError as e:
        if "DLL" in str(e) or "WinError 1114" in str(e):
            logger.warning(
                "torch DLL init failed in-process (%s). "
                "Falling back to subprocess embedding.", e
            )
            return _embed_single_subprocess(text, model_name)
        raise


def _embed_single_subprocess(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed one text in a fresh subprocess that imports torch before sklearn.
    Adds ~1-2s overhead on first call; subsequent calls are fast if model is cached.
    """
    script = (
        "import sys, json, numpy as np\n"
        "import torch\n"
        "from sentence_transformers import SentenceTransformer\n"
        "text = json.loads(sys.argv[1])\n"
        "model_name = sys.argv[2]\n"
        "out_path = sys.argv[3]\n"
        "model = SentenceTransformer(model_name)\n"
        "vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype('float32')\n"
        "np.save(out_path, vec[0])\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as sf:
        sf.write(script)
        script_path = sf.name
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as nf:
        out_path = nf.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path, json.dumps(text), model_name, out_path],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Embedding subprocess failed:\n{proc.stderr[-500:]}"
            )
        return np.load(out_path).astype(np.float32)
    finally:
        Path(script_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


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
