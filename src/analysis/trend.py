"""Day-over-day comparison and trend computation.

Loads the previous day's processed briefing.json and compares
tension index, signal counts, and category shifts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrendData:
    """Trend information computed from day-over-day comparison."""

    previous_composite: float | None = None
    previous_components: dict[str, int] = field(default_factory=dict)
    previous_signal_count: int = 0
    new_signals_delta: int = 0
    category_shifts: dict[str, str] = field(default_factory=dict)
    has_previous: bool = False


def _parse_previous_date(current_date: str) -> str:
    """Compute the previous day's date string."""
    try:
        dt = datetime.strptime(current_date, "%Y-%m-%d").date()
    except ValueError:
        dt = date.today()
    prev = dt - timedelta(days=1)
    return prev.strftime("%Y-%m-%d")


def _load_previous_briefing(
    current_date: str,
    processed_dir: str,
    archive_dir: str,
) -> dict[str, Any] | None:
    """Try to load the previous day's briefing.json.

    Searches in processed dir first, then archive/daily/.

    Args:
        current_date: Current date string (YYYY-MM-DD).
        processed_dir: Path to processed output directory.
        archive_dir: Path to archive directory.

    Returns:
        Parsed briefing dict or None if not found.
    """
    prev_date = _parse_previous_date(current_date)

    # Try processed dir: {processed_dir}/{prev_date}/briefing.json
    paths_to_try = [
        Path(processed_dir) / prev_date / "briefing.json",
        Path(archive_dir) / "daily" / prev_date / "briefing.json",
        Path(archive_dir) / "daily" / f"{prev_date}.json",
    ]

    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load previous briefing from %s: %s", path, exc)

    logger.info("No previous briefing found for %s", prev_date)
    return None


def compute_trends(
    current_date: str,
    current_signals: list[dict[str, Any]],
    processed_dir: str = "",
    archive_dir: str = "",
) -> TrendData:
    """Compute day-over-day trends.

    Args:
        current_date: Current analysis date (YYYY-MM-DD).
        current_signals: Today's classified signals.
        processed_dir: Path to processed output directory.
        archive_dir: Path to archive directory.

    Returns:
        TrendData with previous day comparison data.
    """
    trend = TrendData()

    previous = _load_previous_briefing(current_date, processed_dir, archive_dir)
    if previous is None:
        return trend

    trend.has_previous = True

    # Extract previous tension index
    prev_tension = previous.get("tension_index", {})
    trend.previous_composite = prev_tension.get("composite")

    # Extract previous component scores
    for component in prev_tension.get("components", []):
        name = component.get("name", {})
        en_name = name.get("en", "").lower() if isinstance(name, dict) else ""
        if en_name:
            trend.previous_components[en_name] = component.get("score", 0)

    # Previous signal count
    prev_signals = previous.get("signals", [])
    trend.previous_signal_count = len(prev_signals)
    trend.new_signals_delta = len(current_signals) - len(prev_signals)

    # Category shifts: compare current vs previous category distribution
    prev_categories: dict[str, int] = {}
    for sig in prev_signals:
        cat = sig.get("category", "")
        prev_categories[cat] = prev_categories.get(cat, 0) + 1

    curr_categories: dict[str, int] = {}
    for sig in current_signals:
        cat = sig.get("category", "")
        curr_categories[cat] = curr_categories.get(cat, 0) + 1

    all_cats = set(prev_categories.keys()) | set(curr_categories.keys())
    for cat in all_cats:
        prev_count = prev_categories.get(cat, 0)
        curr_count = curr_categories.get(cat, 0)
        if curr_count > prev_count:
            trend.category_shifts[cat] = "up"
        elif curr_count < prev_count:
            trend.category_shifts[cat] = "down"
        else:
            trend.category_shifts[cat] = "stable"

    return trend
