"""
src/data/clean.py
─────────────────────────────────────────────────────────────────────────────
Text normalisation for TWCS tweets.

Design decisions:
- Preserve original text in a separate column; never overwrite raw.
- Normalise URLs → <URL> for classification/retrieval (removes irrelevant tokens).
- @mention handling: replace with <USER> to prevent privacy leakage and noise.
- Emojis are PRESERVED — they carry sentiment useful for classification.
- Language detection via langdetect; non-English rows flagged (not deleted).
- Deterministic: no randomness introduced here.
- Short messages (<= acknowledgement_threshold) are flagged, not deleted.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Regex patterns
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_HASHTAG_RE = re.compile(r"#(\w+)")  # preserve text, remove #


def normalise_text(
    text: str,
    *,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    lowercase: bool = False,
    strip_hashtags: bool = False,
) -> str:
    """
    Apply normalisation steps to a single tweet string.
    Returns a cleaned string; the original is not modified.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: normalise unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # Step 2: URL replacement
    if replace_urls:
        text = _URL_RE.sub("<URL>", text)

    # Step 3: @mention replacement
    if replace_mentions:
        text = _MENTION_RE.sub("<USER>", text)

    # Step 4: hashtag normalisation — keep word, remove #
    if strip_hashtags:
        text = _HASHTAG_RE.sub(r"\1", text)

    # Step 5: collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    # Step 6: optional lowercase (off by default — preserve casing for LLM)
    if lowercase:
        text = text.lower()

    return text


def detect_language(text: str) -> str:
    """
    Detect language using langdetect.
    Returns ISO 639-1 code or 'unknown' on failure.
    """
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:
        return "unknown"


def is_acknowledgement(text: str, threshold: int = 15) -> bool:
    """
    Return True if the message is likely a trivial acknowledgement/closing.
    Heuristic: very short text OR matches common closing phrases.
    """
    clean = text.strip().lower()
    if len(clean) <= threshold:
        return True
    _ACK_PHRASES = {
        "thanks", "thank you", "thank you!", "thanks!", "ok", "okay",
        "got it", "np", "no problem", "sure", "understood", "noted",
        "you're welcome", "youre welcome", "great", "perfect", "awesome",
        "sounds good", "will do",
    }
    return clean in _ACK_PHRASES


def clean_dataframe(
    df: pd.DataFrame,
    *,
    min_text_length: int = 5,
    acknowledgement_threshold: int = 15,
    detect_lang: bool = False,
    target_lang: str = "en",
) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to a tweet DataFrame.

    Columns added:
    - text_clean     : normalised text (URL/mention replaced)
    - text_lower     : lowercase version for TF-IDF
    - is_ack         : acknowledgement flag
    - lang           : detected language (if detect_lang=True)
    - is_english     : True if lang == target_lang
    - is_duplicate   : True if tweet_id duplicated

    Rows NEVER deleted silently; flags are used instead.
    Caller decides what to filter.
    """
    out = df.copy()

    # 1. De-duplicate by tweet_id
    out["is_duplicate"] = out.duplicated(subset=["tweet_id"], keep="first")
    n_dups = out["is_duplicate"].sum()
    if n_dups:
        logger.info("Flagged %d duplicate tweet_ids", n_dups)

    # 2. Handle missing text
    out["text"] = out["text"].fillna("").astype(str)

    # 3. Normalised text
    out["text_clean"] = out["text"].apply(
        lambda t: normalise_text(t, replace_urls=True, replace_mentions=True)
    )
    out["text_lower"] = out["text_clean"].str.lower()

    # 4. Flag too-short messages
    out["too_short"] = out["text_clean"].str.len() < min_text_length

    # 5. Acknowledgement flag
    out["is_ack"] = out["text_clean"].apply(
        lambda t: is_acknowledgement(t, acknowledgement_threshold)
    )

    # 6. Language detection (optional — slow if enabled on large frames)
    if detect_lang:
        logger.info("Running language detection (this may be slow)...")
        out["lang"] = out["text_clean"].apply(detect_language)
        out["is_english"] = out["lang"] == target_lang
        n_non_en = (~out["is_english"]).sum()
        logger.info("Non-English rows flagged: %d", n_non_en)
    else:
        out["lang"] = "unknown"
        out["is_english"] = True  # assume English for v1

    logger.info(
        "Cleaning complete. too_short=%d, is_ack=%d, duplicates=%d",
        out["too_short"].sum(),
        out["is_ack"].sum(),
        out["is_duplicate"].sum(),
    )
    return out
