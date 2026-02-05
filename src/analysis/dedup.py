"""Signal deduplication for the analysis pipeline.

Provides within-day and cross-day deduplication to prevent the same
news story from appearing in consecutive briefings.  Four tiers of
matching are used:

  1. URL match: identical source URL after normalisation  → duplicate
  2. Title match: SequenceMatcher ratio ≥ threshold       → duplicate
     (0.80 for English, 0.70 for Chinese — shorter headlines)
  3. Title + body: title ratio in fuzzy range AND body Jaccard ≥ 0.60
                                                           → duplicate
  4. Entity + category: same entities AND same category AND
     body Jaccard ≥ 0.50                                   → duplicate

Signals that fall below all thresholds are considered distinct and kept.

Chinese text handling:
  - Language detected via Unicode range (CJK characters)
  - Chinese stop words excluded from Jaccard comparison
  - Lower title similarity threshold for Chinese (0.70 vs 0.80)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — tuned for news headline lengths (10-20 words)
# ---------------------------------------------------------------------------
TITLE_EXACT_THRESHOLD_EN = 0.85   # Very likely same headline, minor edit
TITLE_EXACT_THRESHOLD_ZH = 0.70   # Lower for Chinese (shorter headlines)
TITLE_FUZZY_LOW = 0.50            # Worth checking body overlap
BODY_JACCARD_THRESHOLD = 0.60     # Same substantive content
ENTITY_BODY_JACCARD_THRESHOLD = 0.50  # Lower threshold when entities match

# Default lookback for cross-day deduplication
DEFAULT_LOOKBACK_DAYS = 3

# Stop words excluded from Jaccard body comparison.  Common words that
# inflate similarity without indicating the same story.
_STOP_WORDS_EN = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "has", "have", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "that", "this",
    "it", "its", "not", "no", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "my", "said", "says", "also",
    "as", "if", "so", "than", "can", "about", "more", "up",
    "out", "into", "over", "after", "new", "two", "one",
})

# Chinese stop words (function words, particles, common verbs)
_STOP_WORDS_ZH = frozenset({
    "的", "了", "是", "在", "和", "与", "对", "为", "将", "被",
    "这", "那", "有", "也", "就", "都", "而", "及", "等", "到",
    "从", "向", "于", "以", "把", "给", "让", "用", "并", "或",
    "但", "却", "又", "所", "其", "之", "此", "某", "该", "各",
    "着", "过", "来", "去", "上", "下", "中", "内", "外", "间",
    "后", "前", "时", "日", "月", "年", "说", "称", "表示", "认为",
    "指出", "强调", "要求", "希望", "可以", "能够", "应该", "需要",
    "进行", "开展", "实施", "推动", "加强", "促进", "支持", "反对",
})

# Combined stop words for mixed-language text
_STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_ZH


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@dataclass
class DedupStats:
    """Statistics from a deduplication pass."""

    total_before: int = 0
    total_after: int = 0
    dropped_url: int = 0
    dropped_title: int = 0
    dropped_title_body: int = 0
    dropped_entity_body: int = 0

    @property
    def total_dropped(self) -> int:
        return (
            self.dropped_url
            + self.dropped_title
            + self.dropped_title_body
            + self.dropped_entity_body
        )


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters (CJK Unified Ideographs)."""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def _detect_language(text: str) -> str:
    """Detect primary language of text.

    Returns:
        "zh" if text contains Chinese characters, "en" otherwise.
    """
    return "zh" if _contains_chinese(text) else "en"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _extract_comparable_text(signal: dict[str, Any]) -> tuple[str, str, str]:
    """Extract title, body, and URL from a signal regardless of format.

    Handles both raw fetcher signals (plain strings) and archived
    processed signals (bilingual dicts).

    Returns:
        Tuple of (title, body, url) as plain strings.
    """
    title = signal.get("title", "")
    if isinstance(title, dict):
        title = title.get("en", "")

    body = (
        signal.get("body_text", "")
        or signal.get("body_snippet", "")
        or signal.get("body", "")
    )
    if isinstance(body, dict):
        body = body.get("en", "")

    url = signal.get("source_url", "") or signal.get("url", "")

    return str(title), str(body), str(url)


def _extract_entities(signal: dict[str, Any]) -> set[str]:
    """Extract entity IDs from a signal.

    Looks in both raw format (entities list) and processed format
    (entity_directory with id fields).

    Returns:
        Set of entity ID strings.
    """
    entities: set[str] = set()

    # Raw format: entities is a list of IDs
    raw_entities = signal.get("entities", [])
    if isinstance(raw_entities, list):
        for e in raw_entities:
            if isinstance(e, str):
                entities.add(e)
            elif isinstance(e, dict) and "id" in e:
                entities.add(e["id"])

    # Processed format: entity_ids list
    entity_ids = signal.get("entity_ids", [])
    if isinstance(entity_ids, list):
        for eid in entity_ids:
            if isinstance(eid, str):
                entities.add(eid)

    return entities


def _extract_category(signal: dict[str, Any]) -> str:
    """Extract category from a signal.

    Returns:
        Category string or empty string if not found.
    """
    category = signal.get("category", "")
    if isinstance(category, dict):
        category = category.get("en", "")
    return str(category).lower()


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Lowercases, strips punctuation, and collapses whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_url(url: str) -> str:
    """Normalize a URL for comparison.

    Strips scheme, trailing slashes, query parameters, and fragment.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.lower().strip())
        clean = urlunparse(("", parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
        return clean.strip("/")
    except Exception:
        return url.lower().strip().rstrip("/")


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------
def title_similarity(a: str, b: str) -> float:
    """Compute similarity between two normalized title strings.

    Uses SequenceMatcher (same algorithm as news_scraper._is_duplicate).

    Returns:
        Float in [0, 1] where 1.0 means identical.
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def body_jaccard(a: str, b: str) -> float:
    """Compute Jaccard similarity between two body texts.

    Tokenizes into word/character sets (excluding stop words),
    then computes |intersection| / |union|.

    For mixed-language text:
      - English: words with 3+ characters
      - Chinese: individual characters (since no word boundaries)
      - Stop words excluded for both languages

    Returns:
        Float in [0, 1] where 1.0 means identical word sets.
    """
    if not a or not b:
        return 0.0

    def _tokenize(text: str) -> set[str]:
        tokens: set[str] = set()

        # Extract English words (3+ chars)
        english_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        tokens.update(english_words - _STOP_WORDS_EN)

        # Extract Chinese characters (excluding stop words)
        for char in text:
            if "\u4e00" <= char <= "\u9fff" and char not in _STOP_WORDS_ZH:
                tokens.add(char)

        return tokens

    set_a = _tokenize(a)
    set_b = _tokenize(b)

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Core duplicate check
# ---------------------------------------------------------------------------
def is_duplicate(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any],
) -> tuple[bool, str]:
    """Check if *signal_a* is a duplicate of *signal_b*.

    Uses language-aware thresholds:
      - Chinese text uses lower title similarity threshold (0.70 vs 0.80)
      - Chinese stop words are excluded from body Jaccard comparison

    Returns:
        Tuple of (is_dup, reason).
        *reason* is one of ``"url"``, ``"title"``, ``"title+body"``,
        ``"entity+body"``, or ``""`` (not a duplicate).
    """
    title_a, body_a, url_a = _extract_comparable_text(signal_a)
    title_b, body_b, url_b = _extract_comparable_text(signal_b)

    # Tier 1: URL exact match
    if url_a and url_b:
        if normalize_url(url_a) == normalize_url(url_b):
            return True, "url"

    # Detect language for threshold selection
    lang_a = _detect_language(title_a + body_a)
    lang_b = _detect_language(title_b + body_b)
    is_chinese = lang_a == "zh" or lang_b == "zh"

    # Choose title threshold based on language
    title_threshold = TITLE_EXACT_THRESHOLD_ZH if is_chinese else TITLE_EXACT_THRESHOLD_EN

    # Tier 2: Title similarity (language-aware threshold)
    norm_a = normalize_text(title_a)
    norm_b = normalize_text(title_b)
    t_sim = title_similarity(norm_a, norm_b)

    if t_sim >= title_threshold:
        return True, "title"

    # Tier 3: Title in fuzzy range + body overlap
    if t_sim >= TITLE_FUZZY_LOW:
        b_sim = body_jaccard(body_a, body_b)
        if b_sim >= BODY_JACCARD_THRESHOLD:
            return True, "title+body"

    # Tier 4: Entity-based dedup — same entities + same category + body overlap
    entities_a = _extract_entities(signal_a)
    entities_b = _extract_entities(signal_b)
    category_a = _extract_category(signal_a)
    category_b = _extract_category(signal_b)

    if entities_a and entities_b and category_a and category_b:
        # Check for significant entity overlap (at least 2 common entities or 50% overlap)
        common_entities = entities_a & entities_b
        entity_overlap = len(common_entities) / min(len(entities_a), len(entities_b))

        if (len(common_entities) >= 2 or entity_overlap >= 0.5) and category_a == category_b:
            b_sim = body_jaccard(body_a, body_b)
            if b_sim >= ENTITY_BODY_JACCARD_THRESHOLD:
                return True, "entity+body"

    return False, ""


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------
def load_recent_signals(
    processed_dir: str,
    archive_dir: str,
    current_date: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[dict[str, Any]]:
    """Load signals from the last *lookback_days* archived briefings.

    Searches processed dir first, then archive/daily/ for each date
    (same fallback logic as ``trend._load_previous_briefing``).

    Args:
        processed_dir: Path to processed output directory.
        archive_dir: Path to archive directory.
        current_date: Today's date string (YYYY-MM-DD).
        lookback_days: How many previous days to load.

    Returns:
        Flat list of signal dicts from previous briefings.
    """
    try:
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    except ValueError:
        logger.warning("Invalid date format for dedup lookback: %s", current_date)
        return []

    all_signals: list[dict[str, Any]] = []

    for offset in range(1, lookback_days + 1):
        prev_dt = current_dt - timedelta(days=offset)
        prev_date = prev_dt.strftime("%Y-%m-%d")

        paths_to_try = [
            Path(processed_dir) / prev_date / "briefing.json",
            Path(archive_dir) / "daily" / prev_date / "briefing.json",
        ]

        for path in paths_to_try:
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        briefing = json.load(f)
                    signals = briefing.get("signals", [])
                    all_signals.extend(signals)
                    logger.info(
                        "Dedup: loaded %d signals from %s (%s)",
                        len(signals), prev_date, path,
                    )
                    break  # Found this date, move to next offset
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Dedup: failed to load %s: %s", path, exc)

    logger.info(
        "Dedup: %d total previous signals from last %d day(s)",
        len(all_signals), lookback_days,
    )
    return all_signals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def deduplicate_signals(
    signals: list[dict[str, Any]],
    previous_signals: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], DedupStats]:
    """Remove duplicate signals from the current batch.

    Performs two passes:
      1. **Within-day** — pairwise comparison of current signals
         (first occurrence is kept).
      2. **Cross-day** — each surviving signal vs. all *previous_signals*.

    Args:
        signals: Current day's raw signals.
        previous_signals: Signals from previous days' briefings.

    Returns:
        Tuple of (filtered signals, dedup statistics).
    """
    stats = DedupStats(total_before=len(signals))
    previous_signals = previous_signals or []

    # --- Pass 1: Within-day dedup ---
    kept: list[dict[str, Any]] = []

    for signal in signals:
        dup_found = False
        for existing in kept:
            is_dup, reason = is_duplicate(signal, existing)
            if is_dup:
                title, _, _ = _extract_comparable_text(signal)
                ex_title, _, _ = _extract_comparable_text(existing)
                logger.debug(
                    "Dedup (same-day, %s): dropped '%s' (matches '%s')",
                    reason, title[:80], ex_title[:80],
                )
                if reason == "url":
                    stats.dropped_url += 1
                elif reason == "title":
                    stats.dropped_title += 1
                elif reason == "entity+body":
                    stats.dropped_entity_body += 1
                else:
                    stats.dropped_title_body += 1
                dup_found = True
                break
        if not dup_found:
            kept.append(signal)

    # --- Pass 2: Cross-day dedup ---
    if previous_signals:
        final: list[dict[str, Any]] = []
        for signal in kept:
            dup_found = False
            for prev in previous_signals:
                is_dup, reason = is_duplicate(signal, prev)
                if is_dup:
                    title, _, _ = _extract_comparable_text(signal)
                    prev_title, _, _ = _extract_comparable_text(prev)
                    logger.debug(
                        "Dedup (cross-day, %s): dropped '%s' "
                        "(matches previous '%s')",
                        reason, title[:80], prev_title[:80],
                    )
                    if reason == "url":
                        stats.dropped_url += 1
                    elif reason == "title":
                        stats.dropped_title += 1
                    elif reason == "entity+body":
                        stats.dropped_entity_body += 1
                    else:
                        stats.dropped_title_body += 1
                    dup_found = True
                    break
            if not dup_found:
                final.append(signal)
        kept = final

    stats.total_after = len(kept)
    logger.info(
        "Dedup: %d → %d signals (-%d: url=%d, title=%d, title+body=%d, entity+body=%d)",
        stats.total_before,
        stats.total_after,
        stats.total_dropped,
        stats.dropped_url,
        stats.dropped_title,
        stats.dropped_title_body,
        stats.dropped_entity_body,
    )
    return kept, stats
