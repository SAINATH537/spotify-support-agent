"""
src/agent.py
─────────────────────────────────────────────────────────────────────────────
Main agent orchestrator — wires all components into a single pipeline.

Pipeline:
  customer message
    → intent classification (SentenceTransformerClassifier)
    → intent-aware FAISS retrieval
    → grounded LLM response generation
    → deterministic escalation policy
    → structured JSON output

Design decisions:
- Agent is stateless per call — no session state maintained.
- Components are loaded once at construction (lazy-loaded on first call).
- All thresholds come from config.yaml, not hard-coded.
- Returns a fully typed dict matching the spec's output schema.
- Logs each step for debugging; log level controlled by config.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class SpotifySupportAgent:
    """
    End-to-end agent for SpotifyCares customer support.

    Usage:
        agent = SpotifySupportAgent()
        result = agent.run("I can't log into my account")
        print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        classifier_path: Optional[str] = None,
        index_path: Optional[str] = None,
    ) -> None:
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.brand = self.cfg["data"]["brand"]
        self.top_k = self.cfg["retrieval"]["top_k"]
        self.min_similarity = self.cfg["retrieval"]["min_similarity"]
        self.intent_filter = self.cfg["retrieval"]["intent_filter"]
        self.intent_fallback_k = self.cfg["retrieval"]["intent_fallback_k"]

        gen_cfg = self.cfg["generation"]
        self.provider = gen_cfg.get("provider", "stub")
        self.gen_model = gen_cfg.get("model", "gemini-1.5-flash")
        self.temperature = gen_cfg.get("temperature", 0.2)
        self.max_tokens = gen_cfg.get("max_output_tokens", 300)

        self._classifier_path = classifier_path or self.cfg["classification"]["model_path"]
        self._index_path = index_path or self.cfg["retrieval"]["index_path"]
        self.config_path = config_path

        # Lazy-loaded components
        self._classifier = None
        self._faiss = None
        self._taxonomy = None
        self._escalation = None
        self._generator = None

    # ── Component loaders (lazy) ─────────────────────────────────────────────

    @property
    def classifier(self):
        if self._classifier is None:
            try:
                from src.intent.baseline import TFIDFLogisticRegression
                self._classifier = TFIDFLogisticRegression.load(self._classifier_path)
                logger.info("TF-IDF LR classifier loaded from %s", self._classifier_path)
            except Exception as e:
                logger.error("Classifier load failed: %s", e)
                raise
        return self._classifier

    @property
    def faiss(self):
        if self._faiss is None:
            from src.retrieval.faiss_store import FAISSStore
            self._faiss = FAISSStore.load(self._index_path)
            logger.info("FAISS index loaded from %s", self._index_path)
        return self._faiss

    @property
    def taxonomy(self):
        if self._taxonomy is None:
            from src.intent.taxonomy import IntentTaxonomy
            self._taxonomy = IntentTaxonomy(self.config_path)
        return self._taxonomy

    @property
    def escalation(self):
        if self._escalation is None:
            from src.escalation.policy import EscalationPolicy
            self._escalation = EscalationPolicy(self.config_path)
        return self._escalation

    @property
    def generator(self):
        if self._generator is None:
            from src.generation.generator import ResponseGenerator
            self._generator = ResponseGenerator(
                provider=self.provider,
                model=self.gen_model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        return self._generator

    # ── Main pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        customer_message: str,
        conversation_context: str = "",
    ) -> dict[str, Any]:
        """
        Run the full support agent pipeline.

        Args:
            customer_message: raw customer tweet text
            conversation_context: prior conversation text (optional)

        Returns:
            {
                "intent": str,
                "intent_confidence": float,
                "retrieved_evidence": list[dict],
                "draft_reply": str,
                "decision": "AUTO_HANDLE" | "ESCALATE",
                "decision_reason": str,
                "risk_flags": list[str],
                "grounding_score": float,
                "latency_ms": float,
            }
        """
        t0 = time.perf_counter()

        from src.data.clean import normalise_text

        # ── Step 1: Normalise input ──────────────────────────────────────────
        clean_msg = normalise_text(
            customer_message, replace_urls=True, replace_mentions=True
        )
        logger.info("[Agent] message='%s...'", clean_msg[:60])

        # ── Step 2: Intent classification ────────────────────────────────────
        try:
            intent, intent_confidence = self.classifier.predict_one(clean_msg)
        except Exception as e:
            logger.warning("[Agent] Classifier failed: %s — defaulting to Other", e)
            intent, intent_confidence = "Other / Ambiguous", 0.0

        intent_desc = self.taxonomy.description_for(intent)
        logger.info("[Agent] intent='%s' conf=%.2f", intent, intent_confidence)

        # ── Step 3: Retrieval ────────────────────────────────────────────────
        try:
            import torch  # must precede sentence_transformers import on Windows
            from src.retrieval.embeddings import embed_single
            query_vec = embed_single(clean_msg, model_name=self.cfg["embedding"]["model_name"])
            retrieved = self.faiss.search(
                query_vec=query_vec,
                top_k=self.top_k,
                intent=intent if self.intent_filter else None,
                fallback_k=self.intent_fallback_k,
                min_similarity=0.0,  # policy applies threshold, not retrieval
            )
        except Exception as e:
            logger.warning("[Agent] Retrieval failed: %s", e)
            retrieved = []

        logger.info("[Agent] retrieved %d evidence items", len(retrieved))

        # ── Step 4: Generate reply ───────────────────────────────────────────
        gen_result = self.generator.generate(
            brand=self.brand,
            customer_message=clean_msg,
            intent=intent,
            intent_description=intent_desc,
            retrieved_evidence=retrieved,
            conversation_context=conversation_context,
        )

        # ── Step 5: Escalation decision ──────────────────────────────────────
        esc = self.escalation.decide(
            intent=intent,
            intent_confidence=intent_confidence,
            retrieved_evidence=retrieved,
            grounding_score=gen_result["grounding_score"],
            generation_failed=gen_result.get("generation_failed", False),
            draft_reply=gen_result["draft_reply"],  # scanned for unsupported actions
        )

        latency = (time.perf_counter() - t0) * 1000

        result = {
            "intent": intent,
            "intent_confidence": round(intent_confidence, 4),
            "retrieved_evidence": [
                {
                    "interaction_id": e.get("interaction_id", ""),
                    "similarity": round(e.get("similarity", 0.0), 4),
                    "customer_message": e.get("customer_message", ""),
                    "brand_response": e.get("brand_response", ""),
                    "resolution_type": e.get("resolution_type", ""),
                    "intent": e.get("intent", ""),
                }
                for e in retrieved[:self.top_k]
            ],
            "draft_reply": gen_result["draft_reply"],
            "grounding_score": gen_result["grounding_score"],
            "decision": esc.decision,
            "decision_reason": esc.decision_reason,
            "risk_flags": esc.risk_flags,
            "latency_ms": round(latency, 1),
        }

        logger.info(
            "[Agent] decision=%s in %.0fms", esc.decision, latency
        )
        return result
