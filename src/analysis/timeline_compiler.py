"""Timeline compiler for aggregating daily briefings into timeline format.

Compiles historical briefing data into timeline JSON for visualization
and historical analysis of Canada-China relations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("analysis.timeline")


def compile_canada_china_timeline(
    archive_dir: str,
    timelines_dir: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Compile Canada-China relationship timeline from daily briefings.

    Args:
        archive_dir: Path to archive/daily/ directory
        timelines_dir: Path to timelines/ directory
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Updated timeline dictionary
    """
    archive_path = Path(archive_dir) / "daily"
    timelines_path = Path(timelines_dir)

    # Load existing timeline
    timeline_file = timelines_path / "canada-china.json"
    if timeline_file.exists():
        with open(timeline_file) as f:
            timeline = json.load(f)
    else:
        timeline = _create_empty_timeline("canada-china")

    # Find all daily briefings
    briefing_dates = sorted([
        d.name for d in archive_path.iterdir()
        if d.is_dir() and (d / "briefing.json").exists()
    ])

    if start_date:
        briefing_dates = [d for d in briefing_dates if d >= start_date]
    if end_date:
        briefing_dates = [d for d in briefing_dates if d <= end_date]

    logger.info("Processing %d briefings for timeline", len(briefing_dates))

    # Track existing event IDs to avoid duplicates
    existing_ids = {e["id"] for e in timeline.get("events", [])}

    # Process each briefing
    new_events = []
    tension_trend = []

    for date_str in briefing_dates:
        briefing_file = archive_path / date_str / "briefing.json"
        try:
            with open(briefing_file) as f:
                briefing = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("Skipping %s: %s", date_str, e)
            continue

        # Extract tension trend data point
        tension_index = briefing.get("tension_index", {})
        if tension_index.get("composite") is not None:
            tension_trend.append({
                "date": date_str,
                "score": tension_index["composite"],
                "level": _get_en(tension_index.get("level", "")),
            })

        # Extract milestone signals as events
        for signal in briefing.get("signals", []):
            # Check if this is a milestone or high-severity signal
            is_milestone = signal.get("is_milestone", False)
            severity = signal.get("severity", "low")

            # Include milestones and critical/high severity signals
            if is_milestone or severity in ("critical", "high"):
                event_id = signal.get("id", f"{date_str}-{signal.get('category', 'unknown')}")

                if event_id in existing_ids:
                    continue

                event = _signal_to_event(signal, date_str)
                new_events.append(event)
                existing_ids.add(event_id)

    # Merge new events into timeline
    timeline["events"] = timeline.get("events", []) + new_events
    timeline["events"].sort(key=lambda e: e.get("date", ""))

    # Update tension trend (merge with existing, dedupe by date)
    existing_trend_dates = {t["date"] for t in timeline.get("tension_trend", [])}
    for t in tension_trend:
        if t["date"] not in existing_trend_dates:
            timeline.setdefault("tension_trend", []).append(t)
    timeline["tension_trend"] = sorted(
        timeline.get("tension_trend", []),
        key=lambda t: t["date"]
    )

    # Update date range
    if briefing_dates:
        if not timeline.get("date_range", {}).get("start"):
            timeline.setdefault("date_range", {})["start"] = briefing_dates[0]
        timeline.setdefault("date_range", {})["end"] = briefing_dates[-1]

    # Update metadata
    timeline["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "source_briefings": len(briefing_dates),
        "total_events": len(timeline.get("events", [])),
        "total_milestones": sum(
            1 for e in timeline.get("events", [])
            if e.get("is_milestone", False)
        ),
    }

    logger.info(
        "Timeline updated: %d events, %d tension points",
        len(timeline.get("events", [])),
        len(timeline.get("tension_trend", [])),
    )

    return timeline


def _signal_to_event(signal: dict[str, Any], briefing_date: str) -> dict[str, Any]:
    """Convert a signal to a timeline event."""
    signal_date = signal.get("date", briefing_date)
    # Handle date display format (may be object with en/zh or string)
    if isinstance(signal_date, dict):
        signal_date = signal_date.get("en", briefing_date)
    # Try to extract ISO date from display format
    if signal_date and len(signal_date) >= 10:
        signal_date = signal_date[:10]
    else:
        signal_date = briefing_date

    return {
        "id": signal.get("id", f"{briefing_date}-event"),
        "date": signal_date,
        "title": _ensure_bilingual(signal.get("title", "")),
        "description": _ensure_bilingual(signal.get("body", "")),
        "category": signal.get("category", "political"),
        "timeline_category": signal.get("timeline_category"),
        "severity": signal.get("severity", "moderate"),
        "is_milestone": signal.get("is_milestone", False),
        "source_signal_id": signal.get("id"),
        "source_briefing_date": briefing_date,
        "tags": _extract_tags(signal),
    }


def _extract_tags(signal: dict[str, Any]) -> list[str]:
    """Extract tags from signal for timeline categorization."""
    tags = []

    # Add category as tag
    if signal.get("category"):
        tags.append(signal["category"])

    # Add entity IDs as tags
    for eid in signal.get("entity_ids", []):
        tags.append(eid)

    # Add severity as tag for critical/high
    severity = signal.get("severity", "")
    if severity in ("critical", "high"):
        tags.append(f"severity-{severity}")

    return tags


def _ensure_bilingual(value: Any) -> dict[str, str]:
    """Ensure value is in bilingual format."""
    if isinstance(value, dict) and "en" in value:
        return {"en": value.get("en", ""), "zh": value.get("zh", "")}
    text = str(value) if value else ""
    return {"en": text, "zh": text}


def _get_en(value: Any) -> str:
    """Extract English text from bilingual or string value."""
    if isinstance(value, dict):
        return value.get("en", "")
    return str(value) if value else ""


def _create_empty_timeline(timeline_id: str) -> dict[str, Any]:
    """Create empty timeline structure."""
    return {
        "id": timeline_id,
        "name": {"en": f"{timeline_id} Timeline", "zh": f"{timeline_id}时间线"},
        "description": {"en": "", "zh": ""},
        "date_range": {"start": None, "end": None},
        "events": [],
        "periods": [],
        "tension_trend": [],
        "metadata": {
            "generated_at": None,
            "source_briefings": 0,
            "total_events": 0,
            "total_milestones": 0,
        },
    }


def write_timeline(timeline: dict[str, Any], timelines_dir: str) -> Path:
    """Write timeline to JSON file.

    Args:
        timeline: Timeline dictionary
        timelines_dir: Path to timelines/ directory

    Returns:
        Path to written file
    """
    timelines_path = Path(timelines_dir)
    timelines_path.mkdir(parents=True, exist_ok=True)

    timeline_id = timeline.get("id", "unknown")
    output_file = timelines_path / f"{timeline_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)

    logger.info("Wrote timeline to %s", output_file)
    return output_file


def mark_signal_as_milestone(
    signal_id: str,
    timeline_category: str | None = None,
    archive_dir: str | None = None,
) -> bool:
    """Mark a signal as a milestone in the archive.

    This is a utility for manually flagging important historical events.

    Args:
        signal_id: Signal ID to mark
        timeline_category: Optional timeline category to assign
        archive_dir: Path to archive directory

    Returns:
        True if signal was found and marked, False otherwise
    """
    if not archive_dir:
        return False

    archive_path = Path(archive_dir) / "daily"

    # Search for signal in all briefings
    for date_dir in archive_path.iterdir():
        if not date_dir.is_dir():
            continue

        briefing_file = date_dir / "briefing.json"
        if not briefing_file.exists():
            continue

        with open(briefing_file) as f:
            briefing = json.load(f)

        for signal in briefing.get("signals", []):
            if signal.get("id") == signal_id:
                signal["is_milestone"] = True
                if timeline_category:
                    signal["timeline_category"] = timeline_category

                with open(briefing_file, "w", encoding="utf-8") as f:
                    json.dump(briefing, f, indent=2, ensure_ascii=False)

                logger.info("Marked signal %s as milestone in %s", signal_id, date_dir.name)
                return True

    logger.warning("Signal %s not found in archive", signal_id)
    return False
