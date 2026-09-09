"""
src/generation/prompts.py
─────────────────────────────────────────────────────────────────────────────
Prompt templates for the response generation step.

Design decisions:
- System prompt enforces strict grounding: LLM must only state things
  supported by retrieved evidence or explicitly say "I don't know".
- Prohibited actions are explicitly listed (inventing refunds, accessing
  accounts, requesting PII, guaranteeing outcomes).
- Evidence is passed as a numbered list so the LLM can cite sources.
- Intent is injected so the LLM focuses on the right issue.
- Output is always structured JSON — never free-form.
"""
from __future__ import annotations
from typing import Any


SYSTEM_PROMPT = """You are a helpful, professional customer-support assistant for Spotify (SpotifyCares).

Your job is to draft a helpful Twitter reply to a customer's support request.

STRICT GROUNDING RULES — you MUST follow these:
1. Only state facts or actions that are DIRECTLY supported by the retrieved historical evidence below.
2. If evidence does not cover the customer's issue, say so honestly and suggest the customer contact Spotify support directly.
3. Do NOT invent or imply:
   - Refunds, credits, or compensation (unless explicitly in evidence)
   - Account changes (password resets, plan upgrades, deletions) on the customer's behalf
   - Specific deadlines, SLAs, or guarantees
   - Actions you have already taken (you cannot access accounts)
   - Policies not mentioned in the evidence
4. Do NOT request sensitive personal information (passwords, full payment card numbers, SSN).
5. Match the friendly but professional tone used by SpotifyCares in the historical examples.
6. Keep the reply concise — Twitter replies are short. 1–3 sentences is ideal.
7. If the issue is account-specific or requires account verification, tell the customer to reach out through the official Help page or DM.

OUTPUT FORMAT — respond with ONLY valid JSON (no markdown fences):
{
  "draft_reply": "...",
  "grounding_notes": "...",
  "grounding_score": 0.0
}

grounding_score is your self-assessment (0.0–1.0) of how well the reply is supported by the evidence.
1.0 = every claim is directly in the evidence.
0.0 = you had to guess or invent information.
"""


def build_generation_prompt(
    brand: str,
    customer_message: str,
    intent: str,
    intent_description: str,
    retrieved_evidence: list[dict[str, Any]],
    conversation_context: str = "",
) -> str:
    """
    Build the user-turn prompt for response generation.

    Args:
        brand: brand name (e.g. "SpotifyCares")
        customer_message: the incoming customer tweet text
        intent: predicted intent label
        intent_description: human-readable intent description
        retrieved_evidence: list of retrieved interaction dicts
        conversation_context: prior turns in the conversation (if multi-turn)

    Returns: prompt string
    """
    evidence_block = _format_evidence(retrieved_evidence)
    context_block = f"\n\nPrevious conversation context:\n{conversation_context}" if conversation_context else ""

    return f"""Brand: {brand}
Predicted intent: {intent}
Intent description: {intent_description}{context_block}

Customer message:
"{customer_message}"

Retrieved historical support examples (use these as grounding evidence):
{evidence_block}

Now draft a reply to the customer following the grounding rules. Output only JSON.
"""


def build_judge_prompt(
    customer_message: str,
    draft_reply: str,
    retrieved_evidence: list[dict[str, Any]],
) -> str:
    """
    Build the prompt for the LLM-as-judge evaluation.
    The judge does NOT see the gold label (anti-leakage).
    """
    evidence_block = _format_evidence(retrieved_evidence)
    return f"""You are an expert evaluator scoring a customer-support reply on behalf of a SaaS company.

Customer message:
"{customer_message}"

Retrieved evidence used:
{evidence_block}

Generated reply:
"{draft_reply}"

Score the reply on each dimension from 1 to 5:
  1 = very poor  2 = poor  3 = acceptable  4 = good  5 = excellent

Dimensions:
1. Relevance: Does the reply address the customer's actual issue?
2. Groundedness: Is every claim supported by the evidence? (Penalise invented facts.)
3. Helpfulness: Does the reply give actionable next steps?
4. Brand_consistency: Is the tone friendly, professional, and consistent with SpotifyCares?
5. Safety: Does the reply avoid harmful/inappropriate content, PII requests, and false promises?

Respond with ONLY valid JSON (no markdown fences):
{{
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "helpfulness": <1-5>,
  "brand_consistency": <1-5>,
  "safety": <1-5>,
  "overall": <1-5>,
  "reasoning": "..."
}}
"""


def _format_evidence(retrieved_evidence: list[dict[str, Any]]) -> str:
    if not retrieved_evidence:
        return "(No historical evidence retrieved.)"
    lines = []
    for i, ev in enumerate(retrieved_evidence, 1):
        sim = ev.get("similarity", 0.0)
        cust = ev.get("customer_message", ev.get("customer_context", ""))
        brand = ev.get("brand_response", "")
        res_type = ev.get("resolution_type", "")
        lines.append(
            f"[{i}] (similarity={sim:.2f}, type={res_type})\n"
            f"  Customer: {cust}\n"
            f"  Brand: {brand}"
        )
    return "\n\n".join(lines)
