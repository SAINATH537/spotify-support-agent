"""
src/intent/taxonomy.py
─────────────────────────────────────────────────────────────────────────────
Intent taxonomy loader and weak-labelling utilities.

The taxonomy is defined in config.yaml — NOT hard-coded here.
This makes it easy to add, rename, merge, or split intents without code changes.

Design decisions:
- Keyword-based weak labelling is ONLY used for:
    (a) generating the annotation template
    (b) quick exploration
  It is NEVER used as a gold label.
- Primary intent determined by keyword density; ties → "Other / Ambiguous".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class IntentTaxonomy:
    """
    Loads and exposes the intent taxonomy from config.yaml.

    Attributes:
        intents     : list of intent config dicts
        label_to_id : {label_string: int}
        id_to_label : {int: label_string}
        high_risk   : set of high-risk intent labels
    """

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.intents: list[dict] = cfg["intents"]
        self.label_to_id: dict[str, int] = {i["label"]: i["id"] for i in self.intents}
        self.id_to_label: dict[int, str] = {i["id"]: i["label"] for i in self.intents}
        self.high_risk: set[str] = {i["label"] for i in self.intents if i.get("high_risk")}
        self.n_classes: int = len(self.intents)
        self.labels: list[str] = [i["label"] for i in self.intents]
        logger.info(
            "Loaded taxonomy: %d intents, %d high-risk: %s",
            self.n_classes,
            len(self.high_risk),
            list(self.high_risk),
        )

    def is_high_risk(self, label: str) -> bool:
        return label in self.high_risk

    def id_for(self, label: str) -> int:
        return self.label_to_id[label]

    def label_for(self, class_id: int) -> str:
        return self.id_to_label[class_id]

    def weak_label(self, text: str) -> str:
        """
        Assign a weak label via keyword matching.
        Returns the label with highest keyword hit count, or "Other / Ambiguous".
        Only used for annotation template generation — not for training.
        """
        text_lower = text.lower()
        best_label = "Other / Ambiguous"
        best_count = 0
        for intent in self.intents:
            if not intent["keywords"]:
                continue
            count = sum(1 for kw in intent["keywords"] if kw in text_lower)
            if count > best_count:
                best_count = count
                best_label = intent["label"]
        return best_label

    def weak_label_batch(self, texts: list[str]) -> list[str]:
        return [self.weak_label(t) for t in texts]

    def description_for(self, label: str) -> str:
        for intent in self.intents:
            if intent["label"] == label:
                return intent.get("description", "")
        return ""

    def intent_descriptions_block(self) -> str:
        """Return a formatted block of intent definitions for prompt injection."""
        lines = ["Intent taxonomy (assign by customer's PRIMARY requested outcome):"]
        for intent in self.intents:
            lines.append(f"  {intent['id']+1}. {intent['label']}: {intent['description']}")
        return "\n".join(lines)

    def to_csv_rows(self) -> list[dict]:
        return [
            {
                "id": i["id"],
                "label": i["label"],
                "high_risk": i["high_risk"],
                "description": i["description"],
                "keywords": "|".join(i["keywords"]),
            }
            for i in self.intents
        ]
