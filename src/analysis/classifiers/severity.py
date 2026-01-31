"""Multi-factor severity scoring algorithm.

Computes a severity level (critical/high/elevated/moderate/low) based on:
  1. Source reliability tier
  2. Escalation/de-escalation keywords
  3. Bilateral directness (Canada-China mentions)
  4. Recency factor
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

SOURCE_TIER_SCORES: dict[str, int] = {
    "official": 4,
    "wire": 3,
    "specialist": 2,
    "media": 1,
}

SEVERITY_THRESHOLDS: list[tuple[int, str]] = [
    (10, "critical"),
    (7, "high"),
    (5, "elevated"),
    (3, "moderate"),
    (0, "low"),
]

# Canada-China bilateral keywords
BILATERAL_KEYWORDS_EN = [
    "canada-china", "canada china", "sino-canadian", "canadian",
    "ottawa", "beijing", "trudeau", "canada",
]
BILATERAL_KEYWORDS_ZH = [
    "\u52A0\u62FF\u5927", "\u52A0\u4E2D", "\u6E25\u592A\u534E",
]


def _keyword_modifier_score(
    text: str,
    severity_modifiers: dict[str, Any],
) -> int:
    """Score text against escalation/de-escalation keyword lists.

    Args:
        text: Combined text to scan.
        severity_modifiers: Dict with escalation, moderate_escalation,
            de_escalation keys, each containing en/zh lists and weight.

    Returns:
        Total modifier score (can be negative for de-escalation).
    """
    text_lower = text.lower()
    score = 0

    for modifier_key in ("escalation", "moderate_escalation", "de_escalation"):
        modifier = severity_modifiers.get(modifier_key, {})
        weight = modifier.get("weight", 0)
        en_keywords = modifier.get("en", [])
        zh_keywords = modifier.get("zh", [])

        matched = False
        for kw in en_keywords:
            if kw.lower() in text_lower:
                matched = True
                break

        if not matched:
            for kw in zh_keywords:
                if kw in text:
                    matched = True
                    break

        if matched:
            score += weight

    return score


def _bilateral_score(text: str) -> int:
    """Score bilateral directness.

    Returns:
        2 if directly mentions Canada-China relationship,
        1 if mentions China generally,
        0 otherwise.
    """
    text_lower = text.lower()

    for kw in BILATERAL_KEYWORDS_EN:
        if kw in text_lower:
            return 2

    for kw in BILATERAL_KEYWORDS_ZH:
        if kw in text:
            return 2

    china_keywords = ["china", "chinese", "beijing", "prc", "\u4E2D\u56FD", "\u5317\u4EAC"]
    for kw in china_keywords:
        if kw.lower() in text_lower or kw in text:
            return 1

    return 0


def _recency_score(signal_date: str, reference_date: date | None = None) -> int:
    """Score based on recency of the signal.

    Args:
        signal_date: Date string (various formats supported).
        reference_date: The reference date for comparison. Defaults to today.

    Returns:
        +1 for today, 0 for this week, -1 for older.
    """
    ref = reference_date or date.today()

    parsed_date: date | None = None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%Y/%m/%d"):
        try:
            parsed_date = datetime.strptime(signal_date.strip(), fmt).date()
            break
        except (ValueError, AttributeError):
            continue

    if parsed_date is None:
        # If we can't parse the date, assume it's recent
        return 0

    delta = (ref - parsed_date).days

    if delta <= 0:
        return 1
    if delta <= 7:
        return 0
    return -1


def compute_severity_score(
    text: str,
    source_tier: str,
    category: str,
    signal_date: str = "",
    severity_modifiers: dict[str, Any] | None = None,
    reference_date: date | None = None,
) -> int:
    """Compute the raw severity score for a signal.

    Args:
        text: Combined title + body text.
        source_tier: Source reliability tier (official/wire/specialist/media).
        category: Signal category.
        signal_date: Date string for the signal.
        severity_modifiers: Keyword modifiers dict.
        reference_date: Reference date for recency scoring.

    Returns:
        Raw integer score.
    """
    score = 0

    # Factor 1: Source reliability
    score += SOURCE_TIER_SCORES.get(source_tier, 1)

    # Factor 2: Escalation keywords
    if severity_modifiers:
        score += _keyword_modifier_score(text, severity_modifiers)

    # Factor 3: Bilateral directness
    score += _bilateral_score(text)

    # Factor 4: Recency
    if signal_date:
        score += _recency_score(signal_date, reference_date)

    return max(score, 0)


def score_to_severity(score: int) -> str:
    """Convert a raw score to a severity level string.

    Args:
        score: Raw integer score.

    Returns:
        Severity level string.
    """
    for threshold, level in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


def classify_severity(
    signal: dict[str, Any],
    source_tier: str,
    category: str,
    severity_modifiers: dict[str, Any] | None = None,
    reference_date: date | None = None,
) -> str:
    """Classify a signal's severity level.

    Args:
        signal: Raw signal dictionary.
        source_tier: Source reliability tier.
        category: Signal category.
        severity_modifiers: Keyword modifier dictionaries.
        reference_date: Reference date for recency scoring.

    Returns:
        Severity level string (critical/high/elevated/moderate/low).
    """
    parts: list[str] = []

    title = signal.get("title", "")
    if isinstance(title, dict):
        parts.append(title.get("en", ""))
        parts.append(title.get("zh", ""))
    elif isinstance(title, str):
        parts.append(title)

    body = signal.get("body", "")
    if isinstance(body, dict):
        parts.append(body.get("en", ""))
        parts.append(body.get("zh", ""))
    elif isinstance(body, str):
        parts.append(body)

    for field_name in ("headline", "summary", "content", "description"):
        val = signal.get(field_name, "")
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, dict):
            parts.append(val.get("en", ""))
            parts.append(val.get("zh", ""))

    combined = " ".join(parts)
    signal_date = signal.get("date", "")

    score = compute_severity_score(
        text=combined,
        source_tier=source_tier,
        category=category,
        signal_date=signal_date,
        severity_modifiers=severity_modifiers,
        reference_date=reference_date,
    )

    return score_to_severity(score)
