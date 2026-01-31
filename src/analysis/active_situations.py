"""Active situation tracker.

Maintains and updates a list of ongoing situations in the
Canada-China relationship based on signal patterns and predefined
situation definitions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Known situation patterns: trigger keywords and their associated metadata
KNOWN_SITUATIONS: list[dict[str, Any]] = [
    {
        "id": "canola_trade_dispute",
        "name": {"en": "Canola Trade Dispute", "zh": "\u6CB9\u83DC\u7C7D\u8D38\u6613\u4E89\u7AEF"},
        "trigger_keywords": ["canola", "oilseed", "\u6CB9\u83DC\u7C7D", "\u83DC\u7C7D"],
        "default_severity": "elevated",
        "start_date": "2019-03-01",
    },
    {
        "id": "tech_decoupling",
        "name": {"en": "Tech Decoupling", "zh": "\u79D1\u6280\u8131\u94A9"},
        "trigger_keywords": [
            "Huawei", "5G ban", "semiconductor", "tech ban",
            "\u534E\u4E3A", "5G\u7981\u4EE4", "\u534A\u5BFC\u4F53",
        ],
        "default_severity": "high",
        "start_date": "2018-12-01",
    },
    {
        "id": "foreign_interference",
        "name": {
            "en": "Foreign Interference Investigation",
            "zh": "\u5916\u56FD\u5E72\u9884\u8C03\u67E5",
        },
        "trigger_keywords": [
            "foreign interference", "CSIS", "interference inquiry",
            "\u5916\u56FD\u5E72\u9884", "\u5E72\u9884\u8C03\u67E5",
        ],
        "default_severity": "high",
        "start_date": "2023-02-01",
    },
    {
        "id": "taiwan_strait_tensions",
        "name": {"en": "Taiwan Strait Tensions", "zh": "\u53F0\u6D77\u7D27\u5F20\u5C40\u52BF"},
        "trigger_keywords": [
            "Taiwan Strait", "Taiwan", "cross-strait", "PLA",
            "\u53F0\u6E7E\u6D77\u5CE1", "\u53F0\u6E7E", "\u4E24\u5CB8",
        ],
        "default_severity": "elevated",
        "start_date": "2022-08-01",
    },
    {
        "id": "rare_earth_controls",
        "name": {"en": "Rare Earth Export Controls", "zh": "\u7A00\u571F\u51FA\u53E3\u7BA1\u5236"},
        "trigger_keywords": [
            "rare earth", "gallium", "germanium", "critical minerals",
            "\u7A00\u571F", "\u9553", "\u9574", "\u5173\u952E\u77FF\u4EA7",
        ],
        "default_severity": "elevated",
        "start_date": "2023-07-01",
    },
    {
        "id": "diplomatic_tensions",
        "name": {"en": "Diplomatic Tensions", "zh": "\u5916\u4EA4\u7D27\u5F20"},
        "trigger_keywords": [
            "ambassador", "expelled", "diplomatic", "persona non grata",
            "\u5927\u4F7F", "\u9A71\u9010", "\u5916\u4EA4",
        ],
        "default_severity": "moderate",
        "start_date": "2018-12-01",
    },
]

SEVERITY_ORDER: dict[str, int] = {
    "critical": 5,
    "high": 4,
    "elevated": 3,
    "moderate": 2,
    "low": 1,
}


@dataclass
class ActiveSituation:
    """An active situation being tracked."""

    name: dict[str, str]
    detail: dict[str, str]
    severity: str
    day_count: int = 0
    deadline: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching schema."""
        result: dict[str, Any] = {
            "name": self.name,
            "detail": self.detail,
            "severity": self.severity,
        }
        if self.day_count > 0:
            result["day_count"] = self.day_count
        if self.deadline:
            result["deadline"] = self.deadline
        return result


def _compute_day_count(start_date_str: str, current_date: date) -> int:
    """Compute the number of days since a situation started."""
    try:
        start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        delta = (current_date - start).days
        return max(delta, 0)
    except (ValueError, TypeError):
        return 0


def _signal_matches_situation(
    signal: dict[str, Any],
    trigger_keywords: list[str],
) -> bool:
    """Check if a signal matches a situation's trigger keywords."""
    parts: list[str] = []

    for field_name in ("title", "body", "headline", "summary"):
        val = signal.get(field_name, "")
        if isinstance(val, dict):
            parts.append(val.get("en", ""))
            parts.append(val.get("zh", ""))
        elif isinstance(val, str):
            parts.append(val)

    text = " ".join(parts)
    text_lower = text.lower()

    for kw in trigger_keywords:
        if kw.lower() in text_lower or kw in text:
            return True

    return False


def _upgrade_severity(current: str, signal_severity: str) -> str:
    """Upgrade situation severity if signal severity is higher."""
    current_val = SEVERITY_ORDER.get(current, 0)
    signal_val = SEVERITY_ORDER.get(signal_severity, 0)
    if signal_val > current_val:
        return signal_severity
    return current


def track_situations(
    signals: list[dict[str, Any]],
    current_date_str: str,
    previous_situations: list[dict[str, Any]] | None = None,
) -> list[ActiveSituation]:
    """Track active situations based on current signals.

    Checks each known situation pattern against today's signals.
    Updates severity if matching signals have higher severity.
    Computes day counts from start dates.

    Args:
        signals: Today's classified signals.
        current_date_str: Current date (YYYY-MM-DD).
        previous_situations: Previous day's situations for continuity.

    Returns:
        List of ActiveSituation objects for situations with matching signals.
    """
    try:
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
    except ValueError:
        current_date = date.today()

    active: list[ActiveSituation] = []

    for situation_def in KNOWN_SITUATIONS:
        trigger_keywords = situation_def.get("trigger_keywords", [])
        matching_signals = [
            s for s in signals if _signal_matches_situation(s, trigger_keywords)
        ]

        if not matching_signals:
            continue

        # Compute severity: start with default, upgrade based on signals
        severity = situation_def.get("default_severity", "moderate")
        for sig in matching_signals:
            severity = _upgrade_severity(severity, sig.get("severity", "low"))

        day_count = _compute_day_count(
            situation_def.get("start_date", ""),
            current_date,
        )

        # Build detail text summarizing matching signals
        signal_count = len(matching_signals)
        detail_en = f"{signal_count} related signal(s) detected today."
        detail_zh = (
            f"\u4ECA\u65E5\u68C0\u6D4B\u5230{signal_count}\u6761\u76F8\u5173\u4FE1\u53F7\u3002"
        )

        active.append(
            ActiveSituation(
                name=situation_def["name"],
                detail={"en": detail_en, "zh": detail_zh},
                severity=severity,
                day_count=day_count,
            )
        )

    # Sort by severity (highest first)
    active.sort(key=lambda s: SEVERITY_ORDER.get(s.severity, 0), reverse=True)

    return active


def load_previous_situations(
    current_date: str,
    archive_dir: str,
) -> list[dict[str, Any]]:
    """Load previous day's active situations from archive.

    Args:
        current_date: Current date (YYYY-MM-DD).
        archive_dir: Path to archive directory.

    Returns:
        List of situation dicts from previous day, or empty list.
    """
    try:
        dt = datetime.strptime(current_date, "%Y-%m-%d").date()
        prev_date = (dt - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return []

    prev_path = Path(archive_dir) / "daily" / prev_date / "briefing.json"
    if not prev_path.exists():
        return []

    try:
        with open(prev_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("active_situations", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load previous situations: %s", exc)
        return []
