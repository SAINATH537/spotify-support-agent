"""
src/escalation/policy.py
─────────────────────────────────────────────────────────────────────────────
Deterministic, rule-based escalation layer.

Design decisions:
- The LLM is NOT asked whether to escalate. This is a hard policy engine.
- Rules are evaluated in priority order: high-risk intent first, then
  confidence, retrieval similarity, grounding score, evidence count.
- All thresholds are loaded from config.yaml — no magic numbers in code.
- Risk flags are collected for transparency (shown in API response).
- AUTO_HANDLE is only reached if ALL rules pass.
- This design makes the system safe to modify: changing a threshold in
  config.yaml immediately changes behaviour in production.

Escalation signals (all from config.yaml):
  1. Intent is in high_risk_intents → always ESCALATE
  2. intent_confidence < min_intent_confidence → ESCALATE
  3. max retrieval similarity < min_retrieval_similarity → ESCALATE
  4. grounding_score < min_grounding_score → ESCALATE
  5. evidence_count < min_evidence_count → ESCALATE
  6. generation_failed flag → ESCALATE
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .safety import detect_unsupported_actions, SafetyCheckResult

logger = logging.getLogger(__name__)

ESCALATE = "ESCALATE"
AUTO_HANDLE = "AUTO_HANDLE"


@dataclass
class EscalationResult:
    decision: str                   # "ESCALATE" | "AUTO_HANDLE"
    decision_reason: str
    risk_flags: list[str] = field(default_factory=list)


class EscalationPolicy:
    """
    Deterministic escalation policy engine.
    Loads all thresholds from config.yaml at construction.
    """

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        esc_cfg = cfg.get("escalation", {})

        self.high_risk_intents: set[str] = set(esc_cfg.get("high_risk_intents", []))
        self.min_intent_confidence: float = float(esc_cfg.get("min_intent_confidence", 0.55))
        self.min_retrieval_similarity: float = float(esc_cfg.get("min_retrieval_similarity", 0.30))
        self.min_grounding_score: float = float(esc_cfg.get("min_grounding_score", 0.40))
        self.min_evidence_count: int = int(esc_cfg.get("min_evidence_count", 1))

        logger.info(
            "EscalationPolicy loaded: high_risk=%s, conf_thresh=%.2f, "
            "sim_thresh=%.2f, grounding_thresh=%.2f",
            self.high_risk_intents,
            self.min_intent_confidence,
            self.min_retrieval_similarity,
            self.min_grounding_score,
        )

    def decide(
        self,
        intent: str,
        intent_confidence: float,
        retrieved_evidence: list[dict[str, Any]],
        grounding_score: float,
        generation_failed: bool = False,
        draft_reply: str = "",
    ) -> EscalationResult:
        """
        Evaluate all escalation rules and return a decision.

        Args:
            intent: predicted intent label
            intent_confidence: classifier confidence (0.0–1.0)
            retrieved_evidence: list of retrieved interaction dicts
            grounding_score: generation grounding score (0.0–1.0)
            generation_failed: True if the LLM call failed entirely
            draft_reply: generated reply text (for unsupported action scan)

        Returns: EscalationResult with decision, reason, and risk flags
        """
        risk_flags: list[str] = []

        # Rule 0: Generation failed — no usable reply
        if generation_failed:
            return EscalationResult(
                decision=ESCALATE,
                decision_reason="LLM generation failed — cannot produce a safe reply.",
                risk_flags=["generation_failed"],
            )

        # Rule 1: Unsupported action detected in reply (safety-critical)
        if draft_reply:
            safety = detect_unsupported_actions(draft_reply)
            if safety.has_unsupported_action:
                flags = [f"unsupported_action:{v}" for v in safety.violations]
                return EscalationResult(
                    decision=ESCALATE,
                    decision_reason=(
                        f"Reply contains unsupported action claim(s): "
                        f"{', '.join(safety.violations)}. "
                        "Agent cannot actually perform these actions — routing to human."
                    ),
                    risk_flags=flags,
                )
        # Rule 2: High-risk intent — always escalate regardless of confidence
        if intent in self.high_risk_intents:
            risk_flags.append(f"high_risk_intent:{intent}")
            return EscalationResult(
                decision=ESCALATE,
                decision_reason=(
                    f"Intent '{intent}' is classified as high-risk (account security/billing). "
                    "Routing to human agent for safety."
                ),
                risk_flags=risk_flags,
            )

        # Rule 3: Low intent confidence
        if intent_confidence < self.min_intent_confidence:
            risk_flags.append(
                f"low_intent_confidence:{intent_confidence:.2f}<{self.min_intent_confidence}"
            )
            return EscalationResult(
                decision=ESCALATE,
                decision_reason=(
                    f"Intent confidence {intent_confidence:.2f} is below threshold "
                    f"{self.min_intent_confidence}. Message may be ambiguous or multi-intent."
                ),
                risk_flags=risk_flags,
            )

        # Rule 4: Insufficient evidence
        evidence_count = len(retrieved_evidence)
        if evidence_count < self.min_evidence_count:
            risk_flags.append(f"insufficient_evidence:{evidence_count}")
            return EscalationResult(
                decision=ESCALATE,
                decision_reason=(
                    f"Only {evidence_count} historical cases retrieved "
                    f"(minimum: {self.min_evidence_count}). "
                    "Insufficient evidence to ground a safe reply."
                ),
                risk_flags=risk_flags,
            )

        # Rule 5: Low retrieval similarity
        max_similarity = max(
            (e.get("similarity", 0.0) for e in retrieved_evidence), default=0.0
        )
        if max_similarity < self.min_retrieval_similarity:
            risk_flags.append(
                f"low_retrieval_similarity:{max_similarity:.2f}<{self.min_retrieval_similarity}"
            )
            return EscalationResult(
                decision=ESCALATE,
                decision_reason=(
                    f"Best retrieval similarity {max_similarity:.2f} is below threshold "
                    f"{self.min_retrieval_similarity}. No closely matching historical case found."
                ),
                risk_flags=risk_flags,
            )

        # Rule 6: Low grounding score
        if grounding_score < self.min_grounding_score:
            risk_flags.append(
                f"low_grounding_score:{grounding_score:.2f}<{self.min_grounding_score}"
            )
            return EscalationResult(
                decision=ESCALATE,
                decision_reason=(
                    f"Grounding score {grounding_score:.2f} indicates the reply "
                    "may contain claims not supported by historical evidence."
                ),
                risk_flags=risk_flags,
            )

        # All rules passed — safe to auto-handle
        return EscalationResult(
            decision=AUTO_HANDLE,
            decision_reason=(
                f"All checks passed: intent='{intent}' (conf={intent_confidence:.2f}), "
                f"similarity={max_similarity:.2f}, grounding={grounding_score:.2f}."
            ),
            risk_flags=risk_flags,
        )

    def decide_with_overrides(
        self,
        *,
        intent: str,
        intent_confidence: float,
        retrieved_evidence: list[dict[str, Any]],
        grounding_score: float,
        generation_failed: bool = False,
        # Threshold overrides (for threshold sweep experiments)
        min_intent_confidence: Optional[float] = None,
        min_retrieval_similarity: Optional[float] = None,
        min_grounding_score: Optional[float] = None,
    ) -> EscalationResult:
        """Same as decide() but with per-call threshold overrides for threshold sweep."""
        orig = (self.min_intent_confidence, self.min_retrieval_similarity, self.min_grounding_score)
        if min_intent_confidence is not None:
            self.min_intent_confidence = min_intent_confidence
        if min_retrieval_similarity is not None:
            self.min_retrieval_similarity = min_retrieval_similarity
        if min_grounding_score is not None:
            self.min_grounding_score = min_grounding_score
        result = self.decide(intent, intent_confidence, retrieved_evidence, grounding_score, generation_failed)
        self.min_intent_confidence, self.min_retrieval_similarity, self.min_grounding_score = orig
        return result
