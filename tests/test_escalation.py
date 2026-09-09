"""
tests/test_escalation.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the deterministic escalation policy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.escalation.policy import EscalationPolicy, ESCALATE, AUTO_HANDLE

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _make_policy() -> EscalationPolicy:
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    return EscalationPolicy(CONFIG_PATH)


def _good_evidence() -> list[dict]:
    return [{"similarity": 0.75, "interaction_id": "int_001", "brand_response": "Try restarting."}]


# ── High-risk intent ──────────────────────────────────────────────────────────

def test_high_risk_intent_always_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Account / Login / Security",
        intent_confidence=0.99,  # even very high confidence
        retrieved_evidence=_good_evidence(),
        grounding_score=0.9,
    )
    assert result.decision == ESCALATE
    assert "high_risk" in " ".join(result.risk_flags).lower()


def test_billing_always_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Billing / Payment",
        intent_confidence=0.95,
        retrieved_evidence=_good_evidence(),
        grounding_score=0.9,
    )
    assert result.decision == ESCALATE


# ── Low intent confidence ─────────────────────────────────────────────────────

def test_low_confidence_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",
        intent_confidence=0.20,  # well below threshold
        retrieved_evidence=_good_evidence(),
        grounding_score=0.9,
    )
    assert result.decision == ESCALATE
    assert any("confidence" in f.lower() for f in result.risk_flags)


def test_confidence_at_threshold_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",
        intent_confidence=0.54,  # just below 0.55
        retrieved_evidence=_good_evidence(),
        grounding_score=0.9,
    )
    assert result.decision == ESCALATE


def test_confidence_above_threshold_passes():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",
        intent_confidence=0.80,
        retrieved_evidence=_good_evidence(),
        grounding_score=0.9,
    )
    # Should NOT escalate for confidence alone
    assert any("confidence" not in f.lower() for f in result.risk_flags) or result.decision == AUTO_HANDLE


# ── No evidence ───────────────────────────────────────────────────────────────

def test_no_evidence_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",
        intent_confidence=0.80,
        retrieved_evidence=[],  # empty
        grounding_score=0.9,
    )
    assert result.decision == ESCALATE
    assert any("evidence" in f.lower() for f in result.risk_flags)


# ── Low retrieval similarity ──────────────────────────────────────────────────

def test_low_similarity_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",
        intent_confidence=0.80,
        retrieved_evidence=[{"similarity": 0.05, "interaction_id": "int_001"}],
        grounding_score=0.9,
    )
    assert result.decision == ESCALATE
    assert any("similarity" in f.lower() for f in result.risk_flags)


# ── Low grounding score ───────────────────────────────────────────────────────

def test_low_grounding_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",
        intent_confidence=0.80,
        retrieved_evidence=_good_evidence(),
        grounding_score=0.10,  # very low
    )
    assert result.decision == ESCALATE
    assert any("grounding" in f.lower() for f in result.risk_flags)


# ── Generation failure ────────────────────────────────────────────────────────

def test_generation_failed_escalates():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",
        intent_confidence=0.80,
        retrieved_evidence=_good_evidence(),
        grounding_score=0.9,
        generation_failed=True,
    )
    assert result.decision == ESCALATE
    assert "generation_failed" in result.risk_flags


# ── Happy path — all checks pass ──────────────────────────────────────────────

def test_all_checks_pass_auto_handle():
    policy = _make_policy()
    result = policy.decide(
        intent="Playback / Audio",  # not high-risk
        intent_confidence=0.85,
        retrieved_evidence=[{"similarity": 0.80, "interaction_id": "int_001"}],
        grounding_score=0.75,
        generation_failed=False,
    )
    assert result.decision == AUTO_HANDLE
    assert len(result.risk_flags) == 0


# ── Structured output ─────────────────────────────────────────────────────────

def test_result_has_reason():
    policy = _make_policy()
    result = policy.decide("Playback / Audio", 0.85, _good_evidence(), 0.75)
    assert isinstance(result.decision_reason, str)
    assert len(result.decision_reason) > 0


def test_result_fields():
    policy = _make_policy()
    result = policy.decide("Playback / Audio", 0.85, _good_evidence(), 0.75)
    assert hasattr(result, "decision")
    assert hasattr(result, "decision_reason")
    assert hasattr(result, "risk_flags")
    assert result.decision in (ESCALATE, AUTO_HANDLE)
