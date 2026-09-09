"""
src/data/conversations.py
─────────────────────────────────────────────────────────────────────────────
Reconstruct customer↔SpotifyCares conversation threads.

Design decisions:
- Two-turn conversations (customer → brand) are the primary unit for retrieval.
- Multi-turn chains are supported for context building but kept separate.
- Brand response type is classified to filter trivial messages from the
  FAISS index (DMs, "thanks", "will do", etc. pollute retrieval quality).
- Resolution types are heuristic labels (NOT gold) — easy to override.
- conversation_id = smallest tweet_id in the thread for determinism.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Resolution type heuristics ────────────────────────────────────────────────
# Patterns matched against the brand's cleaned reply text (lowercase).
_RESOLUTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_escalation", re.compile(
        r"(dm us|direct message|check your dm|message us|send us a (dm|private|message)|"
        r"contact.*support|reach.*us.*at|help\.spotify\.com/contact)",
        re.I,
    )),
    ("closing", re.compile(
        r"^(thanks?( for (reaching|getting|contacting))?[.!]?\s*)?"
        r"(let us know|feel free|here if|anything else|hope (that|this) help|"
        r"good luck|stay safe|have a (great|good)|enjoy)[.!\s]*$",
        re.I,
    )),
    ("acknowledgement", re.compile(
        r"^(thanks?[.!]?|thank you[.!]?|ok[.!]?|okay[.!]?|got it[.!]?|"
        r"noted[.!]?|understood[.!]?|sure[.!]?|no problem[.!]?|np[.!]?)$",
        re.I,
    )),
    ("clarification_request", re.compile(
        r"(can you (tell|share|confirm|let us know|send)|"
        r"could you (provide|clarify|confirm)|"
        r"what (device|account|version|error)|"
        r"which (device|plan|country)|"
        r"do you (have|see|get)|more (info|detail|information))",
        re.I,
    )),
    ("informational", re.compile(
        r"(you can|to (do|fix|resolve|enable|disable|access)|"
        r"(here|this) is how|follow these|check out|visit|go to|"
        r"available (at|on|in)|spotify\.com|help\.spotify)",
        re.I,
    )),
    ("troubleshooting", re.compile(
        r"(try (to|a |the )?(restart|reinstall|log.?out|clear|update|uninstall)|"
        r"(restart|reinstall|log.?out|clear cache|update) (the )?app|"
        r"(clean|fresh) (install|reinstall)|"
        r"turn.*off.*on|sign.*out.*back|reset|troubleshoot)",
        re.I,
    )),
]


def classify_resolution(text: str) -> str:
    """
    Classify a brand reply into a resolution type using heuristic patterns.
    Order matters: first match wins.
    """
    t = text.strip().lower()
    for res_type, pattern in _RESOLUTION_PATTERNS:
        if pattern.search(t):
            return res_type
    return "other"


def is_retrieval_worthy(resolution_type: str) -> bool:
    """Return True if this resolution type should enter the FAISS index."""
    return resolution_type in {"troubleshooting", "informational", "clarification_request"}


# ── Dataclass definitions ─────────────────────────────────────────────────────

@dataclass
class Message:
    speaker: str         # "customer" | "brand"
    tweet_id: str
    timestamp: str
    text: str
    text_clean: str


@dataclass
class Conversation:
    conversation_id: str
    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [asdict(m) for m in self.messages],
        }

    def customer_context(self) -> str:
        """All customer messages concatenated (for multi-turn context)."""
        return " | ".join(
            m.text_clean for m in self.messages if m.speaker == "customer"
        )

    def customer_message(self) -> str:
        """The most recent customer message."""
        customer_msgs = [m for m in self.messages if m.speaker == "customer"]
        return customer_msgs[-1].text_clean if customer_msgs else ""

    def brand_response(self) -> str:
        """The most recent brand response."""
        brand_msgs = [m for m in self.messages if m.speaker == "brand"]
        return brand_msgs[-1].text_clean if brand_msgs else ""


@dataclass
class Interaction:
    """
    Single customer–brand interaction record used for retrieval indexing.
    """
    interaction_id: str
    conversation_id: str
    customer_context: str
    customer_message: str
    brand_response: str
    brand_response_raw: str
    resolution_type: str
    retrieval_worthy: bool
    customer_tweet_id: str
    brand_tweet_id: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Core reconstruction logic ─────────────────────────────────────────────────

def reconstruct_pairs(
    brand_df: pd.DataFrame,
    customer_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join brand tweets to the customer tweet they replied to.

    Returns a DataFrame with columns:
      customer_tweet_id, customer_text, customer_text_clean, customer_timestamp,
      brand_tweet_id, brand_text, brand_text_clean, brand_timestamp, resolution_type
    """
    brand = brand_df.copy()
    customer = customer_df.copy()

    # Brand rows must have in_response_to_tweet_id to know which customer they answered
    brand = brand[brand["in_response_to_tweet_id"].notna()].copy()
    brand["in_response_to_tweet_id"] = brand["in_response_to_tweet_id"].astype(int)

    # Build lookup: tweet_id → customer row
    cust_lookup = customer.set_index("tweet_id")

    rows = []
    for _, brow in brand.iterrows():
        cust_id = int(brow["in_response_to_tweet_id"])
        if cust_id not in cust_lookup.index:
            continue
        crow = cust_lookup.loc[cust_id]
        brand_text_clean = brow.get("text_clean", brow.get("text", ""))
        rows.append({
            "customer_tweet_id": int(crow.name),
            "customer_text": crow.get("text", ""),
            "customer_text_clean": crow.get("text_clean", crow.get("text", "")),
            "customer_timestamp": crow.get("created_at", ""),
            "brand_tweet_id": int(brow["tweet_id"]),
            "brand_text": brow.get("text", ""),
            "brand_text_clean": brand_text_clean,
            "brand_timestamp": brow.get("created_at", ""),
            "resolution_type": classify_resolution(brand_text_clean),
        })

    pairs_df = pd.DataFrame(rows)
    pairs_df["retrieval_worthy"] = pairs_df["resolution_type"].apply(is_retrieval_worthy)
    logger.info(
        "Reconstructed %d customer–brand pairs. Retrieval-worthy: %d",
        len(pairs_df),
        pairs_df["retrieval_worthy"].sum(),
    )
    return pairs_df


def build_conversations(pairs_df: pd.DataFrame) -> list[Conversation]:
    """Build Conversation objects from pairs DataFrame."""
    convs = []
    for _, row in pairs_df.iterrows():
        conv_id = f"conv_{int(row['customer_tweet_id'])}_{int(row['brand_tweet_id'])}"
        conv = Conversation(
            conversation_id=conv_id,
            messages=[
                Message(
                    speaker="customer",
                    tweet_id=str(int(row["customer_tweet_id"])),
                    timestamp=str(row["customer_timestamp"]),
                    text=str(row["customer_text"]),
                    text_clean=str(row["customer_text_clean"]),
                ),
                Message(
                    speaker="brand",
                    tweet_id=str(int(row["brand_tweet_id"])),
                    timestamp=str(row["brand_timestamp"]),
                    text=str(row["brand_text"]),
                    text_clean=str(row["brand_text_clean"]),
                ),
            ],
        )
        convs.append(conv)
    logger.info("Built %d conversation objects", len(convs))
    return convs


def build_interactions(
    pairs_df: pd.DataFrame,
    intent_map: Optional[dict[int, str]] = None,
) -> list[Interaction]:
    """
    Build flat Interaction records suitable for retrieval indexing.
    Only retrieval-worthy interactions are included.
    """
    interactions = []
    worthy = pairs_df[pairs_df["retrieval_worthy"] == True].copy()
    for i, row in enumerate(worthy.itertuples(index=False)):
        ctx = row.customer_text_clean
        interactions.append(Interaction(
            interaction_id=f"int_{i:06d}",
            conversation_id=f"conv_{int(row.customer_tweet_id)}_{int(row.brand_tweet_id)}",
            customer_context=ctx,
            customer_message=ctx,
            brand_response=row.brand_text_clean,
            brand_response_raw=row.brand_text,
            resolution_type=row.resolution_type,
            retrieval_worthy=True,
            customer_tweet_id=str(int(row.customer_tweet_id)),
            brand_tweet_id=str(int(row.brand_tweet_id)),
            timestamp=str(row.customer_timestamp),
        ))
    logger.info("Built %d retrieval-worthy interaction records", len(interactions))
    return interactions


def interactions_to_dataframe(interactions: list[Interaction]) -> pd.DataFrame:
    return pd.DataFrame([i.to_dict() for i in interactions])


def conversations_to_jsonl(convs: list[Conversation], path: str) -> None:
    """Write conversations to a JSONL file."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for conv in convs:
            f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d conversations to %s", len(convs), path)
