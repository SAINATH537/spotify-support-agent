"""
src/generation/generator.py
─────────────────────────────────────────────────────────────────────────────
LLM response generator with structured JSON output.

Supports:
  - Google Gemini (recommended — free tier, structured output)
  - OpenAI (optional fallback)
  - Stub (no API key needed — returns template reply for testing)

Design decisions:
- Always parse structured JSON output. If parsing fails, fall back to stub
  and flag the response for escalation.
- temperature=0.2 for low creativity / high grounding.
- grounding_score comes from the LLM's own self-assessment AND a simple
  lexical overlap check (safer double-validation).
- Never let a JSON parse failure silently pass — always return a valid struct.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_generation_prompt, build_judge_prompt

load_dotenv()
logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    return json.loads(text)


def _lexical_grounding_score(reply: str, evidence: list[dict]) -> float:
    """
    Simple lexical grounding check: what fraction of reply content words
    appear in at least one piece of evidence.
    Supplements the LLM's self-reported grounding_score.
    """
    if not evidence:
        return 0.0
    evidence_text = " ".join(
        (e.get("brand_response", "") + " " + e.get("customer_message", "")).lower()
        for e in evidence
    )
    reply_words = set(re.findall(r"\b\w{4,}\b", reply.lower()))
    ev_words = set(re.findall(r"\b\w{4,}\b", evidence_text))
    if not reply_words:
        return 0.0
    return len(reply_words & ev_words) / len(reply_words)


class ResponseGenerator:
    """
    Generates grounded support replies using an LLM.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 300,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY not set. Set it in .env or use provider=stub."
                )
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(
                self.model,
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "stub":
            self._client = "stub"
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def generate(
        self,
        brand: str,
        customer_message: str,
        intent: str,
        intent_description: str,
        retrieved_evidence: list[dict[str, Any]],
        conversation_context: str = "",
    ) -> dict[str, Any]:
        """
        Generate a structured support reply.

        Returns:
            {
                "draft_reply": str,
                "grounding_score": float,   # 0.0–1.0
                "grounding_notes": str,
                "generation_failed": bool,
            }
        """
        prompt = build_generation_prompt(
            brand=brand,
            customer_message=customer_message,
            intent=intent,
            intent_description=intent_description,
            retrieved_evidence=retrieved_evidence,
            conversation_context=conversation_context,
        )

        try:
            raw = self._call_llm(prompt)
            result = _extract_json(raw)
        except Exception as e:
            logger.warning("LLM generation/parse failed: %s", e)
            return self._stub_response(customer_message, retrieved_evidence, failed=True)

        # Validate required fields
        draft = result.get("draft_reply", "").strip()
        if not draft:
            return self._stub_response(customer_message, retrieved_evidence, failed=True)

        llm_grounding = float(result.get("grounding_score", 0.5))
        lex_grounding = _lexical_grounding_score(draft, retrieved_evidence)
        # Weighted average: trust LLM's self-score, validate with lexical
        final_grounding = 0.6 * llm_grounding + 0.4 * lex_grounding

        return {
            "draft_reply": draft,
            "grounding_score": round(final_grounding, 3),
            "grounding_notes": result.get("grounding_notes", ""),
            "generation_failed": False,
        }

    def judge(
        self,
        customer_message: str,
        draft_reply: str,
        retrieved_evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Run the LLM-as-judge evaluation on a generated reply.
        Returns scores dict or None on failure.
        """
        prompt = build_judge_prompt(customer_message, draft_reply, retrieved_evidence)
        try:
            raw = self._call_llm(prompt)
            return _extract_json(raw)
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)
            return None

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        if self.provider == "gemini":
            response = client.generate_content(prompt)
            return response.text
        elif self.provider == "openai":
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            return resp.choices[0].message.content
        elif self.provider == "stub" or client == "stub":
            return self._stub_llm_response(prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _stub_llm_response(self, prompt: str) -> str:
        """Deterministic stub response for testing without API key."""
        return json.dumps({
            "draft_reply": (
                "Thanks for reaching out! We're sorry to hear you're having trouble. "
                "Please visit help.spotify.com or DM us so we can look into this for you. 🎵"
            ),
            "grounding_notes": "Stub response — no LLM called.",
            "grounding_score": 0.3,
        })

    def _stub_response(
        self,
        customer_message: str,
        evidence: list[dict],
        failed: bool = False,
    ) -> dict[str, Any]:
        return {
            "draft_reply": (
                "Thanks for reaching out! We're sorry to hear you're having trouble. "
                "Please visit help.spotify.com or DM us so we can look into this for you. 🎵"
            ),
            "grounding_score": 0.0 if failed else 0.3,
            "grounding_notes": "Generation failed — stub fallback used." if failed else "Stub.",
            "generation_failed": failed,
        }
