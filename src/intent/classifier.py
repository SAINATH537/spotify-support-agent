"""
src/intent/classifier.py
─────────────────────────────────────────────────────────────────────────────
Sentence-transformer embeddings + Logistic Regression classifier.

Design decisions:
- all-MiniLM-L6-v2: 384-dim, 22M params, fast, no GPU required.
- Embeddings are cached with joblib to avoid recomputing on every run.
- class_weight='balanced' compensates for imbalanced intent distribution.
- Returns (label, confidence) — confidence is used by the escalation policy.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _cache_key(texts: list[str], model_name: str) -> str:
    h = hashlib.md5(("||".join(texts) + model_name).encode()).hexdigest()[:16]
    return h


class SentenceTransformerClassifier:
    """
    Intent classifier based on sentence-transformer embeddings + LogisticRegression.

    Usage:
        clf = SentenceTransformerClassifier()
        clf.fit(train_texts, train_labels)
        label, conf = clf.predict_one("I can't log into my account")
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        cache_dir: str = "data/processed/embedding_cache",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._encoder: Optional[object] = None  # sentence_transformers.SentenceTransformer
        self.lr = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced",
            multi_class="multinomial",
            solver="lbfgs",
        )
        self.le = LabelEncoder()
        self.classes_: list[str] = []

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer model: %s", self.model_name)
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed texts with disk cache."""
        cache_key = _cache_key(texts, self.model_name)
        cache_path = self.cache_dir / f"emb_{cache_key}.npy"
        if cache_path.exists():
            logger.debug("Loading embeddings from cache: %s", cache_path)
            return np.load(cache_path)

        encoder = self._get_encoder()
        logger.info("Computing embeddings for %d texts...", len(texts))
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        np.save(cache_path, embeddings)
        logger.info("Embeddings cached to %s", cache_path)
        return embeddings

    def fit(self, X: list[str], y: list[str]) -> "SentenceTransformerClassifier":
        y_enc = self.le.fit_transform(y)
        self.classes_ = list(self.le.classes_)
        embeddings = self._embed(X)
        self.lr.fit(embeddings, y_enc)
        logger.info(
            "SentenceTransformerClassifier fitted. %d examples, %d classes.",
            len(X), len(self.classes_),
        )
        return self

    def predict(self, X: list[str]) -> list[str]:
        embeddings = self._embed(X)
        y_enc = self.lr.predict(embeddings)
        return list(self.le.inverse_transform(y_enc))

    def predict_proba(self, X: list[str]) -> np.ndarray:
        embeddings = self._embed(X)
        return self.lr.predict_proba(embeddings)

    def predict_one(self, text: str) -> tuple[str, float]:
        """Returns (label, confidence). Used by the escalation policy."""
        proba = self.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        return self.classes_[idx], float(proba[idx])

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text — used by the retrieval module at inference."""
        encoder = self._get_encoder()
        vec = encoder.encode([text], normalize_embeddings=True)
        return vec[0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Don't pickle the heavy encoder — reload from HuggingFace on load
        encoder_tmp = self._encoder
        self._encoder = None
        joblib.dump(self, path)
        self._encoder = encoder_tmp
        logger.info("SentenceTransformerClassifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SentenceTransformerClassifier":
        obj = joblib.load(path)
        logger.info("SentenceTransformerClassifier loaded from %s", path)
        return obj
