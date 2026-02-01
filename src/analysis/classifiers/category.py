"""Keyword dictionary-based category classification.

Scores text against each category's keyword list (EN + ZH) and
assigns the highest-scoring category. Tie-breaking prefers more
specific (fewer keyword) categories.
"""

from __future__ import annotations

import re
from typing import Any

# Categories ordered by specificity (fewer keywords = more specific).
# Used as a tiebreaker when two categories have the same score.
SPECIFICITY_ORDER = [
    "legal",
    "social",
    "economic",
    "military",
    "technology",
    "diplomatic",
    "trade",
    "political",  # least specific — only wins ties when no other category matches
]

VALID_CATEGORIES = frozenset(
    ["diplomatic", "trade", "military", "technology", "political", "economic", "social", "legal"]
)


def _score_text_against_keywords(
    text: str,
    keywords: list[str],
    *,
    exact_weight: int = 3,
    partial_weight: int = 0,
) -> int:
    """Score a text against a list of keywords.

    Exact match (case-insensitive word boundary) scores higher than
    a partial (substring) match.

    Args:
        text: The text to score.
        keywords: List of keywords to match against.
        exact_weight: Points for an exact-word match.
        partial_weight: Points for a substring match.

    Returns:
        Total score for this keyword list.
    """
    text_lower = text.lower()
    # Build a set of words for exact matching
    words = set(text_lower.split())
    score = 0

    for keyword in keywords:
        kw_lower = keyword.lower()
        # Check exact word match first (for single-word keywords)
        if " " not in kw_lower:
            if kw_lower in words:
                score += exact_weight
                continue
        # Multi-word keyword or phrase match
        if kw_lower in text_lower:
            score += exact_weight
            continue
        # Partial match: check if any word in the keyword appears
        kw_parts = kw_lower.split()
        for part in kw_parts:
            if len(part) > 2 and part in text_lower:
                score += partial_weight
                break

    return score


def _fallback_category(text: str) -> str:
    """Heuristic fallback when no keyword dictionary scores any hits.

    Checks for financial signals, military terms, and technology terms
    before defaulting to "political".
    """
    t = text.lower()
    # Financial signals: dollar amounts or large numbers + business words
    if re.search(r'\$[\d,.]+[bmk]?\b|\d+\s*(?:billion|million|percent|%)', t):
        if any(w in t for w in (
            'company', 'firm', 'corp', 'inc', 'group',
            'stock', 'share', 'market', 'revenue',
            'profit', 'loss', 'earn', 'sales',
            'price', 'investor', 'ipo', 'fund',
        )):
            return "economic"
    # Military signals
    if any(w in t for w in (
        'military', 'army', 'navy', 'pla', 'missile',
        'defense', 'defence', 'warfare', 'troops',
        'warship', 'fighter jet', 'bomber',
    )):
        return "military"
    # Technology signals
    if any(w in t for w in (
        'chip', 'semiconductor', 'ai ', 'artificial intelligence',
        'cyber', '5g', 'quantum', 'robot',
    )):
        return "technology"
    return "political"


def classify_category(
    text: str,
    categories_dict: dict[str, dict[str, list[str]]],
) -> str:
    """Classify text into one of 8 categories based on keyword matching.

    Args:
        text: Combined title + body text to classify.
        categories_dict: Mapping of category name to {"en": [...], "zh": [...]}.

    Returns:
        Category string (e.g. "diplomatic", "trade", etc.).
        Defaults to "political" if no keywords match.
    """
    scores: dict[str, int] = {}

    for category, lang_keywords in categories_dict.items():
        if category not in VALID_CATEGORIES:
            continue
        en_keywords = lang_keywords.get("en", [])
        zh_keywords = lang_keywords.get("zh", [])

        en_score = _score_text_against_keywords(text, en_keywords)
        zh_score = _score_text_against_keywords(text, zh_keywords)
        scores[category] = en_score + zh_score

    if not scores or max(scores.values()) == 0:
        return _fallback_category(text)

    max_score = max(scores.values())
    # Tie-breaking: prefer more specific categories (fewer total keywords)
    top_categories = [cat for cat, sc in scores.items() if sc == max_score]

    if len(top_categories) == 1:
        return top_categories[0]

    # Use specificity order for tiebreaking
    for cat in SPECIFICITY_ORDER:
        if cat in top_categories:
            return cat

    return top_categories[0]


def classify_signal(
    signal: dict[str, Any],
    categories_dict: dict[str, dict[str, list[str]]],
) -> str:
    """Classify a raw signal dict into a category.

    Extracts text from signal's title and body fields (both EN and ZH)
    and runs keyword classification.

    Args:
        signal: Raw signal dictionary with title, body fields.
        categories_dict: Category keyword dictionaries.

    Returns:
        Category string.
    """
    parts: list[str] = []

    # Handle bilingual title
    title = signal.get("title", "")
    if isinstance(title, dict):
        parts.append(title.get("en", ""))
        parts.append(title.get("zh", ""))
    elif isinstance(title, str):
        parts.append(title)

    # Handle bilingual body — check body, body_text, body_snippet
    for body_key in ("body", "body_text", "body_snippet"):
        body = signal.get(body_key, "")
        if isinstance(body, dict):
            parts.append(body.get("en", ""))
            parts.append(body.get("zh", ""))
        elif isinstance(body, str) and body:
            parts.append(body)
            break  # use best available body field

    # Handle headline (raw fetcher format)
    headline = signal.get("headline", "")
    if isinstance(headline, str):
        parts.append(headline)

    # Handle summary/content
    for field_name in ("summary", "content", "description"):
        val = signal.get(field_name, "")
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, dict):
            parts.append(val.get("en", ""))
            parts.append(val.get("zh", ""))

    combined = " ".join(parts)
    return classify_category(combined, categories_dict)
