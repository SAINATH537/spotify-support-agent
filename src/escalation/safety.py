"""
src/escalation/safety.py
─────────────────────────────────────────────────────────────────────────────
Safety check for unsupported action claims in generated replies.

The LLM MUST NOT claim to have performed account-level actions it cannot
actually execute (e.g. "I refunded you", "I cancelled your subscription",
"I accessed your account"). This module detects such claims deterministically
via regex patterns and returns risk flags.

Design decisions:
- Deterministic regex — fast, auditable, no additional LLM call.
- Patterns cover past-tense ("I refunded") and present-perfect ("I have
  cancelled") forms, both first-person singular and plural.
- If ANY unsupported action is detected the caller should escalate.
- The list of patterns is conservative: prefer fewer false positives over
  false negatives. Add patterns carefully.

Reference: Hiver assignment spec §16.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Unsupported action patterns ───────────────────────────────────────────────
# Match: "I refunded", "I've refunded", "we've cancelled", "I have reset", etc.
# Patterns are case-insensitive.

_UNSUPPORTED_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("refund_issued", re.compile(
        r"\b(i|we)'?ve?\s+(already\s+)?(issued|processed|made|completed|initiated|applied)\s+(a\s+)?(full\s+)?"
        r"refund|i\s+(have|had|refunded\s+you|issued\s+a\s+refund)\b",
        re.I,
    )),
    ("subscription_cancelled", re.compile(
        r"\b(i|we)'?ve?\s+(already\s+)?(cancelled|canceled|terminated|ended)\s+(your\s+)?"
        r"(subscription|premium|plan|account)|i\s+(have\s+)?(cancelled|canceled)\b",
        re.I,
    )),
    ("account_accessed", re.compile(
        r"\b(i|we)'?ve?\s+(already\s+)?(accessed|checked|looked\s+into|viewed|logged\s+into)\s+(your\s+)?account"
        r"|i\s+can\s+see\s+(in\s+)?your\s+account",
        re.I,
    )),
    ("password_reset", re.compile(
        r"\b(i|we)'?ve?\s+(already\s+)?(reset|changed|updated)\s+(your\s+)?password"
        r"|i\s+have\s+reset\s+your\b",
        re.I,
    )),
    ("credit_issued", re.compile(
        r"\b(i|we)'?ve?\s+(already\s+)?(issued|added|applied|credited|given)\s+(you\s+)?"
        r"(a\s+)?(credit|credits)|i\s+(have\s+)?credited\s+your",
        re.I,
    )),
    ("account_changed", re.compile(
        r"\b(i|we)'?ve?\s+(already\s+)?(updated|changed|modified|fixed)\s+(your\s+)?account"
        r"|i\s+have\s+(updated|fixed)\s+your\s+(account|settings|plan|subscription)",
        re.I,
    )),
]


@dataclass
class SafetyCheckResult:
    has_unsupported_action: bool
    violations: list[str] = field(default_factory=list)
    flagged_phrases: list[str] = field(default_factory=list)


def detect_unsupported_actions(reply: str) -> SafetyCheckResult:
    """
    Scan a generated reply for unsupported action claims.

    Args:
        reply: The generated draft reply text.

    Returns:
        SafetyCheckResult — if has_unsupported_action is True, the caller
        should escalate and NOT auto-handle.
    """
    if not reply or not reply.strip():
        return SafetyCheckResult(has_unsupported_action=False)

    violations = []
    flagged_phrases = []

    for action_name, pattern in _UNSUPPORTED_PATTERNS:
        matches = pattern.findall(reply)
        if matches:
            violations.append(action_name)
            # Capture the matched sentence for logging
            for m in re.finditer(pattern, reply):
                # Return up to 80 chars around the match for context
                start = max(0, m.start() - 20)
                end = min(len(reply), m.end() + 40)
                flagged_phrases.append(reply[start:end].strip())

    return SafetyCheckResult(
        has_unsupported_action=len(violations) > 0,
        violations=violations,
        flagged_phrases=flagged_phrases,
    )
