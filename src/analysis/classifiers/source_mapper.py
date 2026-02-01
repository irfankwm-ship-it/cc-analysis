"""Source name to reliability tier mapping.

Maps known source names (EN and ZH) to one of four reliability tiers:
  - official: Government agencies and official bodies
  - wire: International wire services
  - specialist: Think tanks and specialist analysis
  - media: General media outlets
"""

from __future__ import annotations

from typing import Any

SOURCE_TIERS: dict[str, dict[str, list[str]]] = {
    "official": {
        "en": [
            "Global Affairs Canada",
            "PMO",
            "State Council",
            "MOFCOM",
            "PBOC",
            "MFA",
            "Taiwan Ministry of National Defense",
            "Xinhua",
            "CAC",
            "SAMR",
        ],
        "zh": [
            "\u52A0\u62FF\u5927\u5168\u7403\u4E8B\u52A1\u90E8",
            "\u603B\u7406\u529E\u516C\u5BA4",
            "\u56FD\u52A1\u9662",
            "\u5546\u52A1\u90E8",
            "\u4E2D\u56FD\u4EBA\u6C11\u94F6\u884C",
            "\u5916\u4EA4\u90E8",
            "\u53F0\u6E7E\u56FD\u9632\u90E8",
            "\u65B0\u534E\u793E",
            "\u7F51\u4FE1\u529E",
            "\u5E02\u573A\u76D1\u7BA1\u603B\u5C40",
        ],
    },
    "wire": {
        "en": ["Reuters", "AP", "AFP", "Bloomberg", "Nikkei Asia"],
        "zh": [
            "\u8DEF\u900F\u793E",
            "\u7F8E\u8054\u793E",
            "\u6CD5\u65B0\u793E",
            "\u5F6D\u535A",
        ],
    },
    "specialist": {
        "en": [
            "CSIS", "Sinocism", "China Brief", "MERICS", "OSINT",
            "The Diplomat", "Asia Times",
        ],
        "zh": [
            "\u52A0\u62FF\u5927\u5B89\u5168\u60C5\u62A5\u5C40",
            "\u5F00\u6E90\u60C5\u62A5",
        ],
    },
    "media": {
        "en": [
            "Globe and Mail",
            "CBC",
            "South China Morning Post",
            "SCMP",
            "SCMP Politics",
            "SCMP Diplomacy",
            "SCMP Economy",
            "SCMP Business",
            "SCMP Tech",
            "SCMP Geopolitics",
            "BBC",
        ],
        "zh": [
            "\u73AF\u7403\u90AE\u62A5",
            "CBC",
            "\u5357\u534E\u65E9\u62A5",
        ],
    },
}

# Build a flat lookup from source name → tier for fast matching
_SOURCE_LOOKUP: dict[str, str] = {}
for _tier, _lang_map in SOURCE_TIERS.items():
    for _lang, _names in _lang_map.items():
        for _name in _names:
            _SOURCE_LOOKUP[_name.lower()] = _tier


def map_source_tier(source_name: str) -> str:
    """Map a source name to its reliability tier.

    Performs case-insensitive matching against known source names.
    Falls back to substring matching if exact match fails.

    Args:
        source_name: The source name string (EN or ZH).

    Returns:
        Tier string: "official", "wire", "specialist", or "media".
        Defaults to "media" for unknown sources.
    """
    if not source_name:
        return "media"

    # Exact match (case-insensitive)
    tier = _SOURCE_LOOKUP.get(source_name.lower())
    if tier:
        return tier

    # Substring matching: check if any known source name is contained
    source_lower = source_name.lower()
    for known_name, tier in _SOURCE_LOOKUP.items():
        if known_name in source_lower or source_lower in known_name:
            return tier

    return "media"


def map_signal_source_tier(signal: dict[str, Any]) -> str:
    """Extract source name from a signal and map to tier.

    Handles both string and bilingual dict source fields.

    Args:
        signal: Signal dictionary with a 'source' field.

    Returns:
        Source tier string.
    """
    source = signal.get("source", "")

    if isinstance(source, dict):
        # Try English first, then Chinese
        en_source = source.get("en", "")
        tier = map_source_tier(en_source)
        if tier != "media" or not source.get("zh"):
            return tier
        return map_source_tier(source.get("zh", ""))

    if isinstance(source, str):
        return map_source_tier(source)

    return "media"
