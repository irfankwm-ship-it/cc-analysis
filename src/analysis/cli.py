"""CLI entry point for the analysis pipeline.

Provides two commands:
  - analysis run: Full analysis pipeline for a date
  - analysis compile-volume: Compile monthly volume
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import click

from analysis import __version__
from analysis.active_situations import track_situations
from analysis.classifiers.category import classify_signal
from analysis.classifiers.severity import classify_severity
from analysis.classifiers.source_mapper import map_signal_source_tier
from analysis.config import PROJECT_ROOT, load_config
from analysis.entities import build_entity_directory, match_entities_across_signals
from analysis.output import assemble_briefing, validate_briefing, write_archive, write_processed
from analysis.tension_index import compute_tension_index
from analysis.trend import compute_trends
from analysis.volume_compiler import compile_volume, write_volume

logger = logging.getLogger("analysis")


def _setup_logging(level: str, fmt: str) -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stderr,
    )


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _to_bilingual(value: Any) -> dict[str, str]:
    """Ensure a value is in bilingual {"en": ..., "zh": ...} format."""
    if isinstance(value, dict) and "en" in value:
        return value
    text = str(value) if value else ""
    return {"en": text, "zh": text}


def _normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Normalize a classified signal to conform to the processed schema.

    Converts plain string fields to bilingual format and adds
    default values for any missing required fields.
    """
    s = dict(signal)

    # Bilingual text fields
    for key in ("title", "body", "source"):
        if key in s:
            s[key] = _to_bilingual(s[key])
        else:
            s[key] = {"en": "", "zh": ""}

    # Date: keep as string
    if "date" not in s:
        s["date"] = ""

    # Implications: add default if missing
    if "implications" not in s or not isinstance(s["implications"], dict):
        s["implications"] = {
            "canada_impact": {"en": "Assessment pending.", "zh": "评估进行中。"},
            "what_to_watch": {"en": "Monitoring.", "zh": "持续关注。"},
        }
    else:
        imp = s["implications"]
        if "canada_impact" not in imp:
            imp["canada_impact"] = {"en": "Assessment pending.", "zh": "评估进行中。"}
        else:
            imp["canada_impact"] = _to_bilingual(imp["canada_impact"])
        if "what_to_watch" in imp:
            imp["what_to_watch"] = _to_bilingual(imp["what_to_watch"])

    return s


def _load_raw_signals(raw_dir: str) -> list[dict[str, Any]]:
    """Load raw signal data from the raw directory.

    Reads all JSON files in the raw directory and extracts signal-like
    items from them (articles, items, signals, etc.).

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        List of raw signal dicts.
    """
    raw_path = Path(raw_dir)
    signals: list[dict[str, Any]] = []

    if not raw_path.exists():
        logger.warning("Raw directory not found: %s", raw_path)
        return signals

    for json_file in sorted(raw_path.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", json_file, exc)
            continue

        # Handle fetcher envelope format: {"metadata": {...}, "data": {...}}
        if isinstance(data, dict) and "data" in data:
            payload = data["data"]
        else:
            payload = data

        # Extract signals from various payload shapes
        if isinstance(payload, list):
            signals.extend(payload)
        elif isinstance(payload, dict):
            # Check for nested signal arrays
            for key in ("signals", "articles", "items", "results"):
                if key in payload and isinstance(payload[key], list):
                    signals.extend(payload[key])
                    break
            else:
                # The dict itself may be a single signal
                if "title" in payload or "headline" in payload:
                    signals.append(payload)

    return signals


def _load_supplementary_data(raw_dir: str) -> dict[str, Any]:
    """Load supplementary data (trade, market, parliament) from raw files.

    Raw fetcher output is in a different shape than the processed schema.
    This function only loads data that already conforms to the processed
    schema (i.e., has the required top-level keys). Raw data that doesn't
    match is skipped so the analysis defaults take over.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        Dict with trade_data, market_data, parliament keys (each may be None).
    """
    raw_path = Path(raw_dir)
    result: dict[str, Any] = {
        "trade_data": None,
        "market_data": None,
        "parliament": None,
    }

    # Required keys that distinguish processed-format data from raw-format data
    required_keys: dict[str, set[str]] = {
        "trade_data": {"summary_stats", "commodities"},
        "market_data": {"indices", "sectors", "movers", "ipos"},
        "parliament": {"bills", "hansard"},
    }

    file_mapping = {
        "statcan.json": "trade_data",
        "trade.json": "trade_data",
        "yahoo_finance.json": "market_data",
        "market.json": "market_data",
        "parliament.json": "parliament",
    }

    for filename, key in file_mapping.items():
        file_path = raw_path / filename
        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                # Handle fetcher envelope
                if isinstance(data, dict) and "data" in data:
                    payload = data["data"]
                else:
                    payload = data
                # Skip payloads that indicate a fetch error
                if isinstance(payload, dict) and "error" in payload:
                    logger.warning("Skipping %s (error: %s)", filename, payload["error"])
                    continue
                # Only use data that has the required processed-schema keys
                if isinstance(payload, dict) and required_keys[key].issubset(payload.keys()):
                    result[key] = payload
                else:
                    missing = required_keys[key] - set(payload.keys()) if isinstance(payload, dict) else required_keys[key]
                    logger.info(
                        "Skipping %s (raw format, missing: %s); using defaults",
                        filename, ", ".join(sorted(missing)),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", file_path, exc)

    return result


def _determine_volume_number(archive_dir: str) -> int:
    """Determine the volume number for today's briefing.

    Checks existing archive for the highest volume number and increments.
    """
    archive_path = Path(archive_dir) / "daily"
    if not archive_path.exists():
        return 1

    max_vol = 0
    for day_dir in archive_path.iterdir():
        briefing_file = day_dir / "briefing.json" if day_dir.is_dir() else day_dir
        if briefing_file.exists() and briefing_file.suffix == ".json":
            try:
                with open(briefing_file, encoding="utf-8") as f:
                    data = json.load(f)
                vol = data.get("volume", 0)
                max_vol = max(max_vol, vol)
            except (json.JSONDecodeError, OSError):
                continue

    return max_vol + 1


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """China Compass analysis pipeline."""


@main.command()
@click.option("--env", type=click.Choice(["dev", "staging", "prod"]), default=None,
              help="Environment (default: dev or CC_ENV)")
@click.option("--date", "target_date", default=None,
              help="Analysis date in YYYY-MM-DD format (default: today)")
@click.option("--raw-dir", default=None,
              help="Raw data directory (default: ../cc-data/raw/{date}/)")
@click.option("--output-dir", default=None,
              help="Output directory (default: ../cc-data/processed/{date}/)")
@click.option("--archive-dir", default=None,
              help="Archive directory (default: ../cc-data/archive/)")
@click.option("--schemas-dir", default=None,
              help="Schemas directory for validation")
def run(
    env: str | None,
    target_date: str | None,
    raw_dir: str | None,
    output_dir: str | None,
    archive_dir: str | None,
    schemas_dir: str | None,
) -> None:
    """Run the full analysis pipeline for a date."""
    # Load configuration
    config = load_config(env=env)
    _setup_logging(config.logging.level, config.logging.format)

    # Resolve date
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    logger.info("Running analysis for %s (env=%s)", target_date, config.env)

    # Resolve paths
    resolved_raw = raw_dir or str(_resolve_path(config.paths.raw_dir) / target_date)
    resolved_output = output_dir or str(_resolve_path(config.paths.processed_dir))
    resolved_archive = archive_dir or str(_resolve_path(config.paths.archive_dir))
    resolved_schemas = schemas_dir if schemas_dir is not None else str(
        _resolve_path(config.paths.schemas_dir)
    )

    # Step 1: Load raw signals
    logger.info("Loading raw signals from %s", resolved_raw)
    raw_signals = _load_raw_signals(resolved_raw)
    logger.info("Loaded %d raw signals", len(raw_signals))

    # Load supplementary data
    supplementary = _load_supplementary_data(resolved_raw)

    # Step 2: Classify signals (category + severity)
    logger.info("Classifying signals...")
    classified_signals: list[dict[str, Any]] = []

    for signal in raw_signals:
        category = classify_signal(signal, config.keywords.categories)
        source_tier = map_signal_source_tier(signal)
        severity = classify_severity(
            signal,
            source_tier=source_tier,
            category=category,
            severity_modifiers=config.keywords.severity_modifiers,
            reference_date=None,
        )

        # Build classified signal
        classified = dict(signal)
        classified["category"] = category
        classified["severity"] = severity

        # Ensure signal has an ID
        if "id" not in classified:
            title = signal.get("title", "")
            if isinstance(title, dict):
                title = title.get("en", "")
            slug = (
                title.lower().replace(" ", "-")[:50]
                if title
                else f"signal-{len(classified_signals)}"
            )
            classified["id"] = slug

        # Normalize to bilingual schema format — raw fetcher data uses
        # plain strings; the processed schema requires {"en": ..., "zh": ...}
        classified = _normalize_signal(classified)

        classified_signals.append(classified)

    logger.info("Classified %d signals", len(classified_signals))

    # Step 3: Compute trends (load previous day data)
    logger.info("Computing trends...")
    trend_data = compute_trends(
        current_date=target_date,
        current_signals=classified_signals,
        processed_dir=resolved_output,
        archive_dir=resolved_archive,
    )

    # Step 4: Compute tension index
    logger.info("Computing tension index...")
    tension = compute_tension_index(
        signals=classified_signals,
        previous_composite=trend_data.previous_composite,
        previous_components=trend_data.previous_components,
        cap_denominator=config.tension.cap_denominator,
    )
    logger.info("Tension index: %.1f (%s)", tension.composite, tension.level["en"])

    # Step 5: Match entities
    logger.info("Matching entities...")
    entity_matches = match_entities_across_signals(
        classified_signals,
        config.keywords.entity_aliases,
    )
    entity_directory = build_entity_directory(entity_matches, config.keywords.entity_aliases)
    logger.info("Matched %d entities", len(entity_directory))

    # Step 6: Track active situations
    logger.info("Tracking active situations...")
    situations = track_situations(
        signals=classified_signals,
        current_date_str=target_date,
    )
    logger.info("Tracking %d active situations", len(situations))

    # Step 7: Determine volume number
    volume_number = _determine_volume_number(resolved_archive)

    # Step 8: Assemble briefing
    logger.info("Assembling briefing (volume %d)...", volume_number)
    briefing = assemble_briefing(
        date=target_date,
        volume=volume_number,
        signals=classified_signals,
        tension_index=tension.to_dict(),
        trade_data=supplementary.get("trade_data"),
        market_data=supplementary.get("market_data"),
        parliament=supplementary.get("parliament"),
        entities=entity_directory,
        active_situations=[s.to_dict() for s in situations],
    )

    # Step 9: Validate
    logger.info("Validating briefing...")
    is_valid = validate_briefing(briefing, schemas_dir=resolved_schemas)
    if not is_valid:
        logger.error("Briefing validation failed!")
        if config.validation.strict:
            raise click.ClickException(
                "Briefing validation failed. Use non-strict mode to proceed."
            )

    # Step 10: Write output
    logger.info("Writing output...")
    processed_path = write_processed(target_date, briefing, resolved_output)
    archive_path = write_archive(target_date, briefing, resolved_archive)

    logger.info("Analysis complete.")
    logger.info("  Processed: %s", processed_path)
    logger.info("  Archive:   %s", archive_path)

    click.echo(f"Analysis complete for {target_date} (volume {volume_number})")
    click.echo(f"  Signals: {len(classified_signals)}")
    click.echo(f"  Tension: {tension.composite:.1f} ({tension.level['en']})")
    click.echo(f"  Output:  {processed_path}")


@main.command("compile-volume")
@click.option("--env", type=click.Choice(["dev", "staging", "prod"]), default=None,
              help="Environment (default: dev or CC_ENV)")
@click.option("--date", "target_date", default=None,
              help="Reference date (compiles previous month). Default: today.")
@click.option("--archive-dir", default=None,
              help="Archive directory (default: ../cc-data/archive/)")
def compile_volume_cmd(
    env: str | None,
    target_date: str | None,
    archive_dir: str | None,
) -> None:
    """Compile monthly volume from daily briefings."""
    config = load_config(env=env)
    _setup_logging(config.logging.level, config.logging.format)

    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    resolved_archive = archive_dir or str(_resolve_path(config.paths.archive_dir))

    logger.info("Compiling volume for month before %s", target_date)

    volume_meta = compile_volume(target_date, resolved_archive)
    output_path = write_volume(volume_meta, resolved_archive)

    click.echo(f"Volume {volume_meta['volume_number']} compiled")
    click.echo(f"  Period: {volume_meta['period_start']} to {volume_meta['period_end']}")
    click.echo(f"  Signals: {volume_meta['signal_count']}")
    click.echo(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
