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


_IMPACT_TEMPLATES: dict[str, dict[str, str]] = {
    "diplomatic": {
        "en": "May affect bilateral diplomatic relations and consular activity between Canada and China.",
        "zh": "可能影响加中双边外交关系和领事活动。",
    },
    "trade": {
        "en": "Could influence Canada-China trade flows, tariffs, or market access for Canadian exporters.",
        "zh": "可能影响加中贸易往来、关税或加拿大出口商的市场准入。",
    },
    "military": {
        "en": "Relevant to regional security dynamics and Canada's Indo-Pacific defence posture.",
        "zh": "与区域安全态势和加拿大印太防务战略相关。",
    },
    "technology": {
        "en": "May impact technology transfer policies, research collaboration, or supply chain security.",
        "zh": "可能影响技术转让政策、科研合作或供应链安全。",
    },
    "political": {
        "en": "Could shape domestic political debate on Canada's China policy.",
        "zh": "可能影响加拿大国内关于对华政策的政治讨论。",
    },
    "economic": {
        "en": "May affect economic conditions relevant to Canadian businesses operating in or with China.",
        "zh": "可能影响与在华或对华经营的加拿大企业相关的经济环境。",
    },
    "social": {
        "en": "Relevant to diaspora communities, academic exchanges, or public opinion on Canada-China ties.",
        "zh": "与侨民社区、学术交流或加中关系舆论相关。",
    },
    "legal": {
        "en": "May affect regulatory frameworks, sanctions compliance, or rule-of-law considerations.",
        "zh": "可能影响监管框架、制裁合规或法治相关议题。",
    },
}

_WATCH_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "critical": {
        "en": {
            "diplomatic": "Watch for emergency diplomatic recalls, sanctions, or retaliatory measures.",
            "trade": "Watch for immediate trade disruptions, emergency tariffs, or export bans.",
            "military": "Watch for escalation signals, military mobilization, or allied coordination.",
            "technology": "Watch for technology blacklists, emergency export controls, or cyber incidents.",
            "political": "Watch for parliamentary emergency debates or executive policy shifts.",
            "economic": "Watch for capital flight, currency intervention, or investment restrictions.",
            "social": "Watch for travel advisories, evacuation notices, or community safety alerts.",
            "legal": "Watch for sanctions designations, asset freezes, or extradition developments.",
        },
        "zh": {
            "diplomatic": "关注紧急外交召回、制裁或报复措施。",
            "trade": "关注即时贸易中断、紧急关税或出口禁令。",
            "military": "关注局势升级信号、军事调动或盟友协调。",
            "technology": "关注技术黑名单、紧急出口管制或网络安全事件。",
            "political": "关注议会紧急辩论或行政政策转变。",
            "economic": "关注资本外流、汇率干预或投资限制。",
            "social": "关注旅行警告、撤离通知或社区安全提醒。",
            "legal": "关注制裁认定、资产冻结或引渡动态。",
        },
    },
    "high": {
        "en": {
            "diplomatic": "Watch for formal protests, ambassador statements, or coalition responses.",
            "trade": "Watch for new tariff announcements, trade investigation launches, or supply chain shifts.",
            "military": "Watch for military exercises, defence pact discussions, or arms sales decisions.",
            "technology": "Watch for entity list additions, research partnership reviews, or data security rules.",
            "political": "Watch for committee hearings, caucus positions, or opposition policy proposals.",
            "economic": "Watch for investment screening decisions, state enterprise activity, or credit actions.",
            "social": "Watch for university partnership reviews, visa policy changes, or diaspora reactions.",
            "legal": "Watch for new legislation, court rulings, or regulatory enforcement actions.",
        },
        "zh": {
            "diplomatic": "关注正式抗议、大使声明或联盟回应。",
            "trade": "关注新关税公告、贸易调查启动或供应链调整。",
            "military": "关注军事演习、防务协议讨论或武器销售决策。",
            "technology": "关注实体清单增补、科研合作审查或数据安全规定。",
            "political": "关注委员会听证、党团立场或反对党政策提案。",
            "economic": "关注投资审查决定、国有企业动态或信贷行动。",
            "social": "关注大学合作审查、签证政策变化或侨民反应。",
            "legal": "关注新立法、法院裁决或监管执法行动。",
        },
    },
    "default": {
        "en": {
            "diplomatic": "Monitor for follow-up statements or policy adjustments.",
            "trade": "Monitor for trade data releases or business community reactions.",
            "military": "Monitor for regional security developments or defence commentary.",
            "technology": "Monitor for industry responses or regulatory guidance updates.",
            "political": "Monitor for parliamentary questions or media coverage trends.",
            "economic": "Monitor for market reactions or economic indicator releases.",
            "social": "Monitor for community responses or institutional announcements.",
            "legal": "Monitor for regulatory updates or compliance guidance.",
        },
        "zh": {
            "diplomatic": "跟踪后续声明或政策调整。",
            "trade": "跟踪贸易数据发布或商界反应。",
            "military": "跟踪区域安全动态或防务评论。",
            "technology": "跟踪行业反应或监管指导更新。",
            "political": "跟踪议会质询或媒体报道趋势。",
            "economic": "跟踪市场反应或经济指标发布。",
            "social": "跟踪社区反应或机构公告。",
            "legal": "跟踪监管动态或合规指导。",
        },
    },
}


def _generate_implications(category: str, severity: str) -> dict[str, Any]:
    """Generate rule-based implications from category and severity."""
    impact = _IMPACT_TEMPLATES.get(category, _IMPACT_TEMPLATES["diplomatic"])

    severity_key = severity if severity in ("critical", "high") else "default"
    watch_tier = _WATCH_TEMPLATES.get(severity_key, _WATCH_TEMPLATES["default"])
    watch_en = watch_tier["en"].get(category, watch_tier["en"]["diplomatic"])
    watch_zh = watch_tier["zh"].get(category, watch_tier["zh"]["diplomatic"])

    return {
        "canada_impact": impact,
        "what_to_watch": {"en": watch_en, "zh": watch_zh},
    }


def _normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Normalize a classified signal to conform to the processed schema.

    Converts plain string fields to bilingual format and generates
    rule-based implications from category + severity.
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

    # Implications: generate from category + severity if missing
    if "implications" not in s or not isinstance(s["implications"], dict):
        s["implications"] = _generate_implications(
            s.get("category", "diplomatic"),
            s.get("severity", "moderate"),
        )
    else:
        imp = s["implications"]
        if "canada_impact" not in imp:
            imp["canada_impact"] = _IMPACT_TEMPLATES.get(
                s.get("category", "diplomatic"),
                _IMPACT_TEMPLATES["diplomatic"],
            )
        else:
            imp["canada_impact"] = _to_bilingual(imp["canada_impact"])
        if "what_to_watch" not in imp or not imp["what_to_watch"]:
            generated = _generate_implications(
                s.get("category", "diplomatic"),
                s.get("severity", "moderate"),
            )
            imp["what_to_watch"] = generated["what_to_watch"]
        else:
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


def _generate_todays_number(
    supplementary: dict[str, Any],
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate today's number from trade data or signal counts."""
    trade = supplementary.get("trade_data")
    if trade and isinstance(trade, dict):
        totals = trade.get("totals") or trade.get("summary_stats") or {}
        imports_val = totals.get("total_imports_cad")
        exports_val = totals.get("total_exports_cad")
        if imports_val and exports_val:
            total = imports_val + exports_val
            if total >= 1000:
                display = f"${total / 1000:.1f}B"
                display_zh = f"{total / 1000:.1f}0亿加元"
            else:
                display = f"${total:.0f}M"
                display_zh = f"{total:.0f}百万加元"
            return {
                "value": {"en": display, "zh": display_zh},
                "description": {
                    "en": "Canada-China monthly bilateral trade volume",
                    "zh": "加中月度双边贸易总额",
                },
            }

    # Fallback: use signal count
    count = len(signals)
    return {
        "value": {"en": str(count), "zh": str(count)},
        "description": {
            "en": "Canada-China signals tracked today",
            "zh": "今日追踪的加中信号数",
        },
    }


def _generate_quote(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the highest-severity signal's title as the quote."""
    severity_rank = {"critical": 0, "high": 1, "elevated": 2, "moderate": 3, "low": 4}
    # Prefer official sources for the quote
    source_rank = {"Global Affairs Canada": 0, "Parliament of Canada": 1}

    best = None
    best_score = (5, 5)  # (severity_rank, source_rank) — lower is better

    for s in signals:
        sev = severity_rank.get(s.get("severity", "low"), 4)
        src_name = s.get("source", "")
        if isinstance(src_name, dict):
            src_name = src_name.get("en", "")
        src = source_rank.get(src_name, 3)
        score = (sev, src)
        if score < best_score:
            best_score = score
            best = s

    if best:
        title = best.get("title", {})
        if isinstance(title, dict):
            en_title = title.get("en", "")
            zh_title = title.get("zh", en_title)
        else:
            en_title = str(title)
            zh_title = en_title

        source = best.get("source", {})
        if isinstance(source, dict):
            en_source = source.get("en", "")
            zh_source = source.get("zh", en_source)
        else:
            en_source = str(source)
            zh_source = en_source

        date_str = best.get("date", "")

        return {
            "text": {
                "en": f"\u201c{en_title}\u201d",
                "zh": f"\u201c{zh_title}\u201d",
            },
            "attribution": {
                "en": f"\u2014 {en_source}, {date_str}" if date_str else f"\u2014 {en_source}",
                "zh": f"\u2014 {zh_source}，{date_str}" if date_str else f"\u2014 {zh_source}",
            },
        }

    return {
        "text": {"en": "", "zh": ""},
        "attribution": {"en": "", "zh": ""},
    }


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

    # Step 7b: Generate today's number and quote
    todays_number = _generate_todays_number(supplementary, classified_signals)
    quote = _generate_quote(classified_signals)

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


if __name__ == "__main__":
    main()
