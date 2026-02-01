"""Signal deduplication for the analysis pipeline.

Provides within-day and cross-day deduplication to prevent the same
news story from appearing in consecutive briefings.  Three tiers of
matching are used:

  1. URL match: identical source URL after normalisation  → duplicate
  2. Title match: SequenceMatcher ratio ≥ 0.80            → duplicate
  3. Title + body: title ratio 0.50–0.80 AND body Jaccard ≥ 0.60
                                                           → duplicate

Signals that fall below all thresholds are considered distinct and kept.
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
TITLE_EXACT_THRESHOLD = 0.80   # Very likely same headline, minor edit
TITLE_FUZZY_LOW = 0.50         # Worth checking body overlap
BODY_JACCARD_THRESHOLD = 0.60  # Same substantive content

# Stop words excluded from Jaccard body comparison.  Common words that
# inflate similarity without indicating the same story.
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "has", "have", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "that", "this",
    "it", "its", "not", "no", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "my", "said", "says", "also",
    "as", "if", "so", "than", "can", "about", "more", "up",
    "out", "into", "over", "after", "new", "two", "one",
})


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

    @property
    def total_dropped(self) -> int:
        return self.dropped_url + self.dropped_title + self.dropped_title_body


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

    Tokenizes into word sets (excluding stop words and short tokens),
    then computes |intersection| / |union|.

    Returns:
        Float in [0, 1] where 1.0 means identical word sets.
    """
    if not a or not b:
        return 0.0

    def _tokenize(text: str) -> set[str]:
        words = set(re.findall(r"\b\w{3,}\b", text.lower()))
        return words - _STOP_WORDS

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

    Returns:
        Tuple of (is_dup, reason).
        *reason* is one of ``"url"``, ``"title"``, ``"title+body"``,
        or ``""`` (not a duplicate).
    """
    title_a, body_a, url_a = _extract_comparable_text(signal_a)
    title_b, body_b, url_b = _extract_comparable_text(signal_b)

    # Tier 1: URL exact match
    if url_a and url_b:
        if normalize_url(url_a) == normalize_url(url_b):
            return True, "url"

    # Tier 2: Title similarity
    norm_a = normalize_text(title_a)
    norm_b = normalize_text(title_b)
    t_sim = title_similarity(norm_a, norm_b)

    if t_sim >= TITLE_EXACT_THRESHOLD:
        return True, "title"

    # Tier 3: Title in fuzzy range + body overlap
    if t_sim >= TITLE_FUZZY_LOW:
        b_sim = body_jaccard(body_a, body_b)
        if b_sim >= BODY_JACCARD_THRESHOLD:
            return True, "title+body"

    return False, ""


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------
def load_recent_signals(
    processed_dir: str,
    archive_dir: str,
    current_date: str,
    lookback_days: int = 3,
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
                    else:
                        stats.dropped_title_body += 1
                    dup_found = True
                    break
            if not dup_found:
                final.append(signal)
        kept = final

    stats.total_after = len(kept)
    logger.info(
        "Dedup: %d → %d signals (-%d: url=%d, title=%d, title+body=%d)",
        stats.total_before,
        stats.total_after,
        stats.total_dropped,
        stats.dropped_url,
        stats.dropped_title,
        stats.dropped_title_body,
    )
    return kept, stats
