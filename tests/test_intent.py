"""
tests/test_intent.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for intent taxonomy, baselines, and classifier interfaces.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent.taxonomy import IntentTaxonomy
from src.intent.baseline import MajorityClassifier, TFIDFLogisticRegression


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


# ── IntentTaxonomy ────────────────────────────────────────────────────────────

def test_taxonomy_loads():
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    tax = IntentTaxonomy(CONFIG_PATH)
    assert tax.n_classes == 10
    assert "Other / Ambiguous" in tax.labels


def test_taxonomy_high_risk():
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    tax = IntentTaxonomy(CONFIG_PATH)
    assert tax.is_high_risk("Account / Login / Security")
    assert tax.is_high_risk("Billing / Payment")
    assert not tax.is_high_risk("Playback / Audio")


def test_taxonomy_weak_label_account():
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    tax = IntentTaxonomy(CONFIG_PATH)
    label = tax.weak_label("I can't log into my account and my password doesn't work")
    assert label == "Account / Login / Security"


def test_taxonomy_weak_label_billing():
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    tax = IntentTaxonomy(CONFIG_PATH)
    label = tax.weak_label("I was charged twice on my credit card this month")
    assert label == "Billing / Payment"


def test_taxonomy_weak_label_other():
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    tax = IntentTaxonomy(CONFIG_PATH)
    label = tax.weak_label("aaaaaaa bbbbbb")
    assert label == "Other / Ambiguous"


def test_taxonomy_roundtrip():
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    tax = IntentTaxonomy(CONFIG_PATH)
    for label in tax.labels:
        id_ = tax.id_for(label)
        assert tax.label_for(id_) == label


# ── MajorityClassifier ────────────────────────────────────────────────────────

SAMPLE_X = [
    "I can't log in", "account locked", "password reset needed",
    "app crashes", "app not working", "crashes on start",
    "I can't log in again", "account issue",
]
SAMPLE_Y = [
    "Account / Login / Security", "Account / Login / Security", "Account / Login / Security",
    "App / Device / Technical", "App / Device / Technical", "App / Device / Technical",
    "Account / Login / Security", "Account / Login / Security",
]


def test_majority_classifier_predict():
    clf = MajorityClassifier()
    clf.fit(SAMPLE_X, SAMPLE_Y)
    preds = clf.predict(["some random text"])
    assert preds[0] == "Account / Login / Security"  # majority class


def test_majority_classifier_predict_one():
    clf = MajorityClassifier()
    clf.fit(SAMPLE_X, SAMPLE_Y)
    label, conf = clf.predict_one("anything")
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0


def test_majority_classifier_save_load():
    clf = MajorityClassifier()
    clf.fit(SAMPLE_X, SAMPLE_Y)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "majority.joblib"
        clf.save(path)
        loaded = MajorityClassifier.load(path)
        assert loaded.predict(["test"]) == clf.predict(["test"])


# ── TFIDFLogisticRegression ───────────────────────────────────────────────────

EXTENDED_X = SAMPLE_X * 5  # sklearn needs enough samples per class
EXTENDED_Y = SAMPLE_Y * 5


def test_tfidf_lr_predict_type():
    clf = TFIDFLogisticRegression(random_state=42)
    clf.fit(EXTENDED_X, EXTENDED_Y)
    preds = clf.predict(EXTENDED_X[:3])
    assert isinstance(preds, list)
    assert all(isinstance(p, str) for p in preds)


def test_tfidf_lr_predict_proba_shape():
    clf = TFIDFLogisticRegression(random_state=42)
    clf.fit(EXTENDED_X, EXTENDED_Y)
    proba = clf.predict_proba(EXTENDED_X[:3])
    assert proba.shape[0] == 3
    assert abs(proba[0].sum() - 1.0) < 1e-5


def test_tfidf_lr_predict_one():
    clf = TFIDFLogisticRegression(random_state=42)
    clf.fit(EXTENDED_X, EXTENDED_Y)
    label, conf = clf.predict_one("app crashes on startup")
    assert label in ["Account / Login / Security", "App / Device / Technical"]
    assert 0.0 <= conf <= 1.0


def test_tfidf_lr_save_load():
    clf = TFIDFLogisticRegression(random_state=42)
    clf.fit(EXTENDED_X, EXTENDED_Y)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tfidf.joblib"
        clf.save(path)
        loaded = TFIDFLogisticRegression.load(path)
        assert loaded.predict(["test"]) == clf.predict(["test"])
