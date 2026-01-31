"""Monthly volume compilation.

Reads all daily briefings from archive/daily/ for a given month
and aggregates them into a volume summary with metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_month_range(reference_date: str) -> tuple[date, date]:
    """Get the start and end dates of the previous month.

    Args:
        reference_date: A date string (YYYY-MM-DD). The previous month
            relative to this date is compiled.

    Returns:
        Tuple of (start_date, end_date) for the previous month.
    """
    try:
        ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
    except ValueError:
        ref = date.today()

    # Go to first day of current month, then back one day for last day of prev month
    first_of_month = ref.replace(day=1)
    last_of_prev = first_of_month - __import__("datetime").timedelta(days=1)
    first_of_prev = last_of_prev.replace(day=1)

    return first_of_prev, last_of_prev


def _load_daily_briefings(
    start_date: date,
    end_date: date,
    archive_dir: str,
) -> list[dict[str, Any]]:
    """Load all daily briefing files within a date range.

    Args:
        start_date: Start of range (inclusive).
        end_date: End of range (inclusive).
        archive_dir: Path to archive directory.

    Returns:
        List of briefing dicts, sorted by date.
    """
    briefings: list[dict[str, Any]] = []
    archive_path = Path(archive_dir) / "daily"

    if not archive_path.exists():
        logger.warning("Archive daily directory not found: %s", archive_path)
        return briefings

    current = start_date
    delta = __import__("datetime").timedelta(days=1)
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")

        # Try directory format: daily/{date}/briefing.json
        file_path = archive_path / date_str / "briefing.json"
        if not file_path.exists():
            # Try flat format: daily/{date}.json
            file_path = archive_path / f"{date_str}.json"

        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                data["_date"] = date_str
                briefings.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load briefing for %s: %s", date_str, exc)

        current += delta

    return briefings


def _compute_next_volume_number(archive_dir: str) -> int:
    """Determine the next volume number from existing volumes.

    Args:
        archive_dir: Path to archive directory.

    Returns:
        Next sequential volume number (starting from 1).
    """
    volumes_path = Path(archive_dir) / "volumes"
    if not volumes_path.exists():
        return 1

    max_vol = 0
    for file_path in volumes_path.glob("vol-*.json"):
        try:
            num_str = file_path.stem.replace("vol-", "")
            num = int(num_str)
            max_vol = max(max_vol, num)
        except ValueError:
            continue

    return max_vol + 1


def compile_volume(
    reference_date: str,
    archive_dir: str,
) -> dict[str, Any]:
    """Compile a monthly volume from daily briefings.

    Reads all daily briefings for the previous month relative to
    reference_date and aggregates them into volume metadata.

    Args:
        reference_date: Date string (YYYY-MM-DD). Previous month is compiled.
        archive_dir: Path to archive directory.

    Returns:
        Volume metadata dict conforming to volume-meta.schema.json.
    """
    start_date, end_date = _get_month_range(reference_date)
    briefings = _load_daily_briefings(start_date, end_date, archive_dir)

    volume_number = _compute_next_volume_number(archive_dir)

    # Aggregate signals
    total_signals = 0
    category_breakdown: dict[str, int] = {}
    severity_breakdown: dict[str, int] = {}
    tension_trend: list[dict[str, Any]] = []

    for briefing in briefings:
        signals = briefing.get("signals", [])
        total_signals += len(signals)

        for signal in signals:
            cat = signal.get("category", "unknown")
            category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

            sev = signal.get("severity", "unknown")
            severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1

        # Tension index trend line
        tension = briefing.get("tension_index", {})
        composite = tension.get("composite")
        if composite is not None:
            tension_trend.append({
                "date": briefing.get("date", briefing.get("_date", "")),
                "value": composite,
            })

    return {
        "volume_number": volume_number,
        "period_start": start_date.strftime("%Y-%m-%d"),
        "period_end": end_date.strftime("%Y-%m-%d"),
        "signal_count": total_signals,
        "tension_trend": tension_trend,
        "category_breakdown": category_breakdown,
        "severity_breakdown": severity_breakdown,
    }


def write_volume(
    volume_meta: dict[str, Any],
    archive_dir: str,
) -> Path:
    """Write volume metadata to archive/volumes/.

    Args:
        volume_meta: Volume metadata dict.
        archive_dir: Path to archive directory.

    Returns:
        Path to the written file.
    """
    volumes_path = Path(archive_dir) / "volumes"
    volumes_path.mkdir(parents=True, exist_ok=True)

    vol_num = volume_meta["volume_number"]
    file_path = volumes_path / f"vol-{vol_num:03d}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(volume_meta, f, ensure_ascii=False, indent=2)

    logger.info("Wrote volume %d to %s", vol_num, file_path)
    return file_path
