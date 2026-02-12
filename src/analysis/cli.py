"""CLI entry point for the analysis pipeline.

Provides commands:
  - analysis run: Full analysis pipeline for a date
  - analysis compile-volume: Compile monthly volume
  - analysis compile-timeline: Compile timeline from daily briefings
  - analysis mark-milestone: Mark a signal as a milestone
"""

from __future__ import annotations

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
from analysis.data_transforms import (
    determine_volume_number,
    extract_market_signals,
    generate_quote,
    generate_todays_number,
    load_supplementary_data,
)
from analysis.dedup import deduplicate_signals, load_recent_signals
from analysis.entities import (
    build_entity_directory,
    match_entities_across_signals,
    match_entities_in_signal,
)
from analysis.output import assemble_briefing, validate_briefing, write_archive, write_processed
from analysis.signal_filtering import (
    filter_and_prioritize_signals,
    filter_low_value_signals,
    is_china_relevant,
    load_raw_signals,
)
from analysis.signal_normalization import (
    is_primarily_chinese,
    normalize_signal,
    translate_signals_batch,
)
from analysis.tension_index import compute_tension_index
from analysis.timeline_compiler import (
    compile_canada_china_timeline,
    mark_signal_as_milestone,
    write_timeline,
)
from analysis.translate import fix_english_text
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
    config = load_config(env=env)
    _setup_logging(config.logging.level, config.logging.format)

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
    raw_signals = load_raw_signals(resolved_raw)
    logger.info("Loaded %d raw signals", len(raw_signals))

    # Filter to recent signals and prioritize bilateral news
    ft = config.thresholds.filtering
    raw_signals = filter_and_prioritize_signals(
        raw_signals, target_date,
        min_signals=ft.min_signals,
        max_signals=ft.max_signals,
        windows_hours=ft.recency_windows_hours,
        max_per_source=ft.max_per_source,
    )

    # China-relevance gate
    pre_count = len(raw_signals)
    raw_signals = [s for s in raw_signals if is_china_relevant(s)]
    if len(raw_signals) < pre_count:
        logger.info(
            "China-relevance filter: dropped %d of %d signals",
            pre_count - len(raw_signals), pre_count,
        )

    # Value filter
    pre_count = len(raw_signals)
    raw_signals = filter_low_value_signals(raw_signals, min_score=0)
    if len(raw_signals) < pre_count:
        logger.info(
            "Value filter: kept %d of %d signals",
            len(raw_signals), pre_count,
        )

    # Pre-classify for dedup
    logger.info("Pre-classifying signals for dedup...")
    for signal in raw_signals:
        if "category" not in signal:
            signal["category"] = classify_signal(signal, config.keywords.categories)
        if "entity_ids" not in signal:
            signal["entity_ids"] = match_entities_in_signal(signal, config.keywords.entity_aliases)

    # Deduplicate
    logger.info("Deduplicating signals...")
    dt = config.thresholds.dedup
    previous_signals = load_recent_signals(
        processed_dir=resolved_output,
        archive_dir=resolved_archive,
        current_date=target_date,
        lookback_days=dt.lookback_days,
    )
    raw_signals, dedup_stats = deduplicate_signals(
        raw_signals, previous_signals,
        title_exact_en=dt.title_exact_en,
        title_exact_zh=dt.title_exact_zh,
        title_fuzzy_low=dt.title_fuzzy_low,
        body_jaccard_threshold=dt.body_jaccard,
        entity_body_jaccard_threshold=dt.entity_body_jaccard,
    )

    # Load supplementary data
    supplementary = load_supplementary_data(resolved_raw)

    # Step 2: Classify signals
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

        classified = dict(signal)
        classified["category"] = category
        classified["severity"] = severity

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

        classified = normalize_signal(
            classified,
            impact_templates=config.templates.impact_templates or None,
            watch_templates=config.templates.watch_templates or None,
            canada_perspective=config.templates.canada_perspective or None,
            china_perspective=config.templates.china_perspective or None,
            source_names=config.chinese_sources.source_names or None,
            domains=config.chinese_sources.domains or None,
            name_translations=config.chinese_sources.name_translations or None,
        )

        classified_signals.append(classified)

    logger.info("Classified %d signals", len(classified_signals))

    # Step 2b: Translate to Chinese
    logger.info("Translating signals to Chinese...")
    tt = config.thresholds.translation
    classified_signals = translate_signals_batch(
        classified_signals,
        body_truncate_chars=tt.body_truncate_chars,
        english_fragment_threshold=tt.english_fragment_threshold,
    )

    # Step 2c: Quality filter
    pre_count = len(classified_signals)
    quality_filtered = []
    for s in classified_signals:
        body_en = s.get("body", {}).get("en", "")
        if not body_en or len(body_en.strip()) < 20:
            title_preview = s.get("title", {}).get("en", "")[:50]
            logger.debug("Dropping signal with empty body: %s", title_preview)
            continue
        title_en = s.get("title", {}).get("en", "")
        if is_primarily_chinese(title_en):
            logger.warning("Dropping signal with untranslated title: %s", title_en[:50])
            continue

        if title_en:
            fixed_title = fix_english_text(title_en)
            if fixed_title != title_en:
                s["title"]["en"] = fixed_title
        if body_en:
            fixed_body = fix_english_text(body_en)
            if fixed_body != body_en:
                s["body"]["en"] = fixed_body

        quality_filtered.append(s)
    classified_signals = quality_filtered
    if len(classified_signals) < pre_count:
        logger.info(
            "Quality filter: dropped %d signals (empty body or translation failure)",
            pre_count - len(classified_signals),
        )

    # Step 3: Compute trends
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

    # Step 7: Supplementary content
    volume_number = determine_volume_number(resolved_archive)
    todays_number = generate_todays_number(supplementary, classified_signals)
    quote = generate_quote(classified_signals)

    market_signals, regulatory_signals = extract_market_signals(classified_signals)
    logger.info(
        "Extracted %d market signals, %d regulatory signals",
        len(market_signals), len(regulatory_signals),
    )

    md = supplementary.get("market_data") or {}
    md["market_signals"] = market_signals
    md["regulatory_signals"] = regulatory_signals

    # Step 8: Assemble briefing
    logger.info("Assembling briefing (volume %d)...", volume_number)
    briefing = assemble_briefing(
        date=target_date,
        volume=volume_number,
        signals=classified_signals,
        tension_index=tension.to_dict(),
        trade_data=supplementary.get("trade_data"),
        market_data=md,
        parliament=supplementary.get("parliament"),
        entities=entity_directory,
        active_situations=[s.to_dict() for s in situations],
        todays_number=todays_number,
        quote_of_the_day=quote,
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


@main.command("compile-timeline")
@click.option("--env", type=click.Choice(["dev", "staging", "prod"]), default=None,
              help="Environment (default: dev or CC_ENV)")
@click.option("--start-date", default=None,
              help="Start date filter (YYYY-MM-DD). Default: all available.")
@click.option("--end-date", default=None,
              help="End date filter (YYYY-MM-DD). Default: today.")
@click.option("--archive-dir", default=None,
              help="Archive directory (default: ../cc-data/archive/)")
@click.option("--timelines-dir", default=None,
              help="Timelines output directory (default: ../cc-data/timelines/)")
def compile_timeline_cmd(
    env: str | None,
    start_date: str | None,
    end_date: str | None,
    archive_dir: str | None,
    timelines_dir: str | None,
) -> None:
    """Compile Canada-China timeline from daily briefings."""
    config = load_config(env=env)
    _setup_logging(config.logging.level, config.logging.format)

    resolved_archive = archive_dir or str(_resolve_path(config.paths.archive_dir))
    resolved_timelines = timelines_dir or str(
        _resolve_path(config.paths.archive_dir).parent / "timelines"
    )

    logger.info("Compiling Canada-China timeline")
    logger.info("  Archive: %s", resolved_archive)
    logger.info("  Output:  %s", resolved_timelines)
    if start_date:
        logger.info("  From:    %s", start_date)
    if end_date:
        logger.info("  To:      %s", end_date)

    timeline = compile_canada_china_timeline(
        archive_dir=resolved_archive,
        timelines_dir=resolved_timelines,
        start_date=start_date,
        end_date=end_date,
    )

    output_path = write_timeline(timeline, resolved_timelines)

    click.echo("Timeline compiled successfully")
    click.echo(f"  Events:    {timeline['metadata']['total_events']}")
    click.echo(f"  Milestones: {timeline['metadata']['total_milestones']}")
    click.echo(f"  Tension points: {len(timeline.get('tension_trend', []))}")
    click.echo(f"  Output:    {output_path}")


@main.command("mark-milestone")
@click.argument("signal_id")
@click.option("--timeline-category", default=None,
              type=click.Choice([
                  "crisis", "escalation", "de-escalation", "agreement",
                  "policy_shift", "leadership", "incident", "sanction", "negotiation"
              ]),
              help="Timeline category for the milestone")
@click.option("--archive-dir", default=None,
              help="Archive directory (default: ../cc-data/archive/)")
@click.option("--env", type=click.Choice(["dev", "staging", "prod"]), default=None,
              help="Environment (default: dev or CC_ENV)")
def mark_milestone_cmd(
    signal_id: str,
    timeline_category: str | None,
    archive_dir: str | None,
    env: str | None,
) -> None:
    """Mark a signal as a historical milestone."""
    config = load_config(env=env)
    _setup_logging(config.logging.level, config.logging.format)

    resolved_archive = archive_dir or str(_resolve_path(config.paths.archive_dir))

    success = mark_signal_as_milestone(
        signal_id=signal_id,
        timeline_category=timeline_category,
        archive_dir=resolved_archive,
    )

    if success:
        click.echo(f"Marked signal '{signal_id}' as milestone")
        if timeline_category:
            click.echo(f"  Category: {timeline_category}")
    else:
        click.echo(f"Signal '{signal_id}' not found in archive", err=True)
        raise click.ClickException("Signal not found")


if __name__ == "__main__":
    main()
