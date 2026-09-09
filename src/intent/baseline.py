"""
src/intent/baseline.py
─────────────────────────────────────────────────────────────────────────────
Two simple baseline classifiers:
1. MajorityClassifier  — always predict most frequent training class.
2. TFIDFLogisticRegression — sklearn Pipeline(TfidfVectorizer, LogisticRegression).

Both expose a sklearn-compatible interface: fit / predict / predict_proba.
This makes it trivial to swap into the same evaluation harness as the
embedding-based classifier.

Design decision:
- Do NOT replace baselines with anything fancier.
- They establish a lower bound and prove the embedding model adds value.
- TF-IDF LR is often surprisingly competitive — documenting this is important.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class MajorityClassifier:
    """Always predicts the most frequent class seen during training."""

    def __init__(self) -> None:
        self._clf = DummyClassifier(strategy="most_frequent")
        self.le = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, X: list[str], y: list[str]) -> "MajorityClassifier":
        y_enc = self.le.fit_transform(y)
        self.classes_ = list(self.le.classes_)
        self._clf.fit([[0]] * len(X), y_enc)
        majority = self.le.inverse_transform([self._clf.predict([[0]])[0]])[0]
        logger.info("MajorityClassifier fitted. Majority class: '%s'", majority)
        return self

    def predict(self, X: list[str]) -> list[str]:
        y_enc = self._clf.predict([[0]] * len(X))
        return list(self.le.inverse_transform(y_enc))

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self._clf.predict_proba([[0]] * len(X))

    def predict_one(self, text: str) -> tuple[str, float]:
        """Returns (label, confidence)."""
        proba = self.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        return self.classes_[idx], float(proba[idx])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("MajorityClassifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MajorityClassifier":
        return joblib.load(path)


class TFIDFLogisticRegression:
    """
    TF-IDF vectoriser + Logistic Regression.
    Operates on raw (cleaned) text strings.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\w{1,}",
            )),
            ("lr", LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                class_weight="balanced",
                multi_class="multinomial",
                solver="lbfgs",
            )),
        ])
        self.le = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, X: list[str], y: list[str]) -> "TFIDFLogisticRegression":
        y_enc = self.le.fit_transform(y)
        self.classes_ = list(self.le.classes_)
        self.pipeline.fit(X, y_enc)
        logger.info(
            "TFIDFLogisticRegression fitted on %d examples, %d classes",
            len(X), len(self.classes_),
        )
        return self

    def predict(self, X: list[str]) -> list[str]:
        y_enc = self.pipeline.predict(X)
        return list(self.le.inverse_transform(y_enc))

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_one(self, text: str) -> tuple[str, float]:
        proba = self.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        return self.classes_[idx], float(proba[idx])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("TFIDFLogisticRegression saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFLogisticRegression":
        return joblib.load(path)
