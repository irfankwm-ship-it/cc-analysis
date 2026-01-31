"""Dictionary-based entity matching.

Scans signal titles and bodies for mentions of known entities
using the entity_aliases.yaml dictionary, and returns matched
entity IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EntityMatch:
    """A matched entity with its canonical ID and mention count."""

    entity_id: str
    mention_count: int = 0
    matched_aliases: list[str] | None = None

    def __post_init__(self) -> None:
        if self.matched_aliases is None:
            self.matched_aliases = []


def _extract_text(signal: dict[str, Any]) -> str:
    """Extract all searchable text from a signal."""
    parts: list[str] = []

    for field_name in ("title", "body", "headline", "summary", "content", "description"):
        val = signal.get(field_name, "")
        if isinstance(val, dict):
            parts.append(val.get("en", ""))
            parts.append(val.get("zh", ""))
        elif isinstance(val, str):
            parts.append(val)

    # Also check implications
    implications = signal.get("implications", {})
    if isinstance(implications, dict):
        for sub_key in ("canada_impact", "what_to_watch"):
            sub = implications.get(sub_key, {})
            if isinstance(sub, dict):
                parts.append(sub.get("en", ""))
                parts.append(sub.get("zh", ""))

    return " ".join(parts)


def match_entities_in_signal(
    signal: dict[str, Any],
    entity_aliases: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Find all entity IDs mentioned in a single signal.

    Args:
        signal: Signal dictionary to scan.
        entity_aliases: Mapping of entity_id -> {"en": [...], "zh": [...]}.

    Returns:
        List of matched entity IDs (deduplicated).
    """
    text = _extract_text(signal)
    text_lower = text.lower()
    matched: set[str] = set()

    for entity_id, lang_aliases in entity_aliases.items():
        en_aliases = lang_aliases.get("en", [])
        zh_aliases = lang_aliases.get("zh", [])

        for alias in en_aliases:
            if alias.lower() in text_lower:
                matched.add(entity_id)
                break

        if entity_id not in matched:
            for alias in zh_aliases:
                if alias in text:
                    matched.add(entity_id)
                    break

    return sorted(matched)


def match_entities_across_signals(
    signals: list[dict[str, Any]],
    entity_aliases: dict[str, dict[str, list[str]]],
) -> list[EntityMatch]:
    """Match entities across all signals and return aggregated results.

    Args:
        signals: List of signal dictionaries to scan.
        entity_aliases: Entity alias dictionary.

    Returns:
        List of EntityMatch objects, sorted by mention count (descending).
    """
    entity_counts: dict[str, int] = {}

    for signal in signals:
        matched_ids = match_entities_in_signal(signal, entity_aliases)
        for eid in matched_ids:
            entity_counts[eid] = entity_counts.get(eid, 0) + 1

    results = [
        EntityMatch(entity_id=eid, mention_count=count)
        for eid, count in entity_counts.items()
    ]
    results.sort(key=lambda e: e.mention_count, reverse=True)

    return results


def build_entity_directory(
    entity_matches: list[EntityMatch],
    entity_aliases: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Build entity directory entries for the briefing output.

    Creates entity objects conforming to entities.schema.json.

    Args:
        entity_matches: List of matched entities.
        entity_aliases: Entity alias dictionary for name lookup.

    Returns:
        List of entity dicts matching the schema.
    """
    entities: list[dict[str, Any]] = []

    # Entity type mapping based on known IDs
    entity_types: dict[str, str] = {
        "xi_jinping": "people",
        "wang_yi": "people",
        "two_michaels": "people",
        "mofcom": "institution",
        "mfa": "institution",
        "csis": "institution",
        "ufwd": "institution",
        "mss": "institution",
        "huawei": "org",
        "canola": "commodity",
        "rare_earths": "commodity",
        "softwood_lumber": "commodity",
    }

    for match in entity_matches:
        eid = match.entity_id
        aliases = entity_aliases.get(eid, {})
        en_names = aliases.get("en", [])
        zh_names = aliases.get("zh", [])

        primary_en = en_names[0] if en_names else eid
        primary_zh = zh_names[0] if zh_names else eid

        entity_type = entity_types.get(eid, "org")

        entities.append({
            "id": eid,
            "name": {"en": primary_en, "zh": primary_zh},
            "type": entity_type,
            "description": {
                "en": f"Mentioned in {match.mention_count} signal(s) today.",
                "zh": (
                    f"\u4ECA\u65E5\u5728{match.mention_count}"
                    "\u6761\u4FE1\u53F7\u4E2D\u88AB\u63D0\u53CA\u3002"
                ),
            },
            "has_detail_page": False,
        })

    return entities
