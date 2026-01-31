"""Output assembly, validation, and writing.

Assembles a complete briefing.json from all analysis components,
validates against the cc-data schema, and writes to the output directories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def assemble_briefing(
    date: str,
    volume: int,
    signals: list[dict[str, Any]],
    tension_index: dict[str, Any],
    trade_data: dict[str, Any] | None = None,
    market_data: dict[str, Any] | None = None,
    parliament: dict[str, Any] | None = None,
    entities: list[dict[str, Any]] | None = None,
    active_situations: list[dict[str, Any]] | None = None,
    quote_of_the_day: dict[str, Any] | None = None,
    todays_number: dict[str, Any] | None = None,
    disruptions: list[dict[str, Any]] | None = None,
    pathway_cards: list[dict[str, Any]] | None = None,
    explore_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a complete briefing envelope.

    Args:
        date: Publication date (YYYY-MM-DD).
        volume: Sequential volume number.
        signals: Classified signal objects.
        tension_index: Tension index data.
        trade_data: Trade data snapshot.
        market_data: Market data snapshot.
        parliament: Parliament tracking data.
        entities: Entity directory entries.
        active_situations: Active situation list.
        quote_of_the_day: Quote data.
        todays_number: Today's number data.
        disruptions: Disruption items.
        pathway_cards: Navigation pathway cards.
        explore_cards: Explore section cards.

    Returns:
        Complete briefing dict conforming to briefing.schema.json.
    """
    briefing: dict[str, Any] = {
        "date": date,
        "volume": volume,
        "signals": signals,
        "tension_index": tension_index,
        "trade_data": trade_data or _default_trade_data(),
        "market_data": market_data or _default_market_data(),
        "parliament": parliament or _default_parliament(),
        "entities": entities or [],
        "active_situations": active_situations or [],
        "quote_of_the_day": quote_of_the_day or _default_quote(),
        "todays_number": todays_number or _default_number(),
        "disruptions": disruptions or [],
    }

    if pathway_cards is not None:
        briefing["pathway_cards"] = pathway_cards
    if explore_cards is not None:
        briefing["explore_cards"] = explore_cards

    return briefing


def validate_briefing(
    briefing: dict[str, Any],
    schemas_dir: str = "",
) -> bool:
    """Validate a briefing against the JSON schema.

    Args:
        briefing: Briefing dict to validate.
        schemas_dir: Path to schemas directory. If empty, skips validation.

    Returns:
        True if valid, False otherwise.
    """
    if not schemas_dir:
        logger.warning("No schemas directory provided; skipping validation.")
        return True

    schemas_path = Path(schemas_dir)
    schema_file = schemas_path / "briefing.schema.json"

    if not schema_file.exists():
        logger.warning("Schema file not found: %s; skipping validation.", schema_file)
        return True

    try:
        from jsonschema import RefResolver, ValidationError, validate

        with open(schema_file, encoding="utf-8") as f:
            schema = json.load(f)

        # Build a local store keyed by each schema's $id so $ref resolution
        # stays local instead of fetching from remote URLs.
        store: dict[str, Any] = {}
        for sf in schemas_path.glob("*.schema.json"):
            with open(sf, encoding="utf-8") as f:
                s = json.load(f)
            sid = s.get("$id", sf.name)
            store[sid] = s
            store[sf.name] = s

        schema_uri = "file:///" + str(schemas_path.resolve()).replace("\\", "/") + "/"
        resolver = RefResolver(schema_uri, schema, store=store)

        validate(instance=briefing, schema=schema, resolver=resolver)
        logger.info("Briefing validation passed.")
        return True

    except ValidationError as exc:
        path_str = " > ".join(str(p) for p in exc.absolute_path)
        logger.error("Briefing validation failed at %s: %s", path_str, exc.message)
        return False

    except ImportError:
        logger.warning("jsonschema not installed; skipping validation.")
        return True

    except Exception as exc:
        logger.error("Validation error: %s", exc)
        return False


def write_processed(
    date: str,
    briefing: dict[str, Any],
    output_dir: str,
) -> Path:
    """Write briefing.json to the processed output directory.

    Creates {output_dir}/{date}/briefing.json.

    Args:
        date: Date string (YYYY-MM-DD).
        briefing: Complete briefing dict.
        output_dir: Base output directory.

    Returns:
        Path to the written file.
    """
    out_path = Path(output_dir) / date
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / "briefing.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(briefing, f, ensure_ascii=False, indent=2)

    logger.info("Wrote processed briefing to %s", file_path)

    # Also write a 'latest' symlink/copy for easy access
    latest_path = Path(output_dir) / "latest"
    latest_path.mkdir(parents=True, exist_ok=True)
    latest_file = latest_path / "briefing.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(briefing, f, ensure_ascii=False, indent=2)

    return file_path


def write_archive(
    date: str,
    briefing: dict[str, Any],
    archive_dir: str,
) -> Path:
    """Write briefing.json to the archive directory.

    Creates {archive_dir}/daily/{date}/briefing.json.

    Args:
        date: Date string (YYYY-MM-DD).
        briefing: Complete briefing dict.
        archive_dir: Base archive directory.

    Returns:
        Path to the written file.
    """
    out_path = Path(archive_dir) / "daily" / date
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / "briefing.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(briefing, f, ensure_ascii=False, indent=2)

    logger.info("Wrote archive briefing to %s", file_path)
    return file_path


def _default_trade_data() -> dict[str, Any]:
    """Return minimal default trade data."""
    return {
        "summary_stats": [],
        "commodities": [],
    }


def _default_market_data() -> dict[str, Any]:
    """Return minimal default market data."""
    return {
        "indices": [],
        "sectors": [],
        "movers": {"gainers": [], "losers": []},
        "ipos": [],
    }


def _default_parliament() -> dict[str, Any]:
    """Return minimal default parliament data."""
    return {
        "bills": [],
        "hansard": {
            "session_mentions": 0,
            "month_mentions": 0,
            "top_topic": {"en": "N/A", "zh": "N/A"},
            "top_topic_pct": "0%",
        },
    }


def _default_quote() -> dict[str, Any]:
    """Return minimal default quote of the day."""
    return {
        "text": {"en": "", "zh": ""},
        "attribution": {"en": "", "zh": ""},
    }


def _default_number() -> dict[str, Any]:
    """Return minimal default today's number."""
    return {
        "value": {"en": "", "zh": ""},
        "description": {"en": "", "zh": ""},
    }
