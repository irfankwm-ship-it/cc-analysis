"""CLI entry point for the analysis pipeline.

Provides two commands:
  - analysis run: Full analysis pipeline for a date
  - analysis compile-volume: Compile monthly volume
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import date, datetime, timedelta
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
        "en": ("May affect bilateral diplomatic relations and "
               "consular activity between Canada and China."),
        "zh": "可能影响加中双边外交关系和领事活动。",
    },
    "trade": {
        "en": ("Could influence Canada-China trade flows, tariffs, "
               "or market access for Canadian exporters."),
        "zh": "可能影响加中贸易往来、关税或加拿大出口商的市场准入。",
    },
    "military": {
        "en": ("Relevant to regional security dynamics and "
               "Canada's Indo-Pacific defence posture."),
        "zh": "与区域安全态势和加拿大印太防务战略相关。",
    },
    "technology": {
        "en": ("May impact technology transfer policies, research "
               "collaboration, or supply chain security."),
        "zh": "可能影响技术转让政策、科研合作或供应链安全。",
    },
    "political": {
        "en": ("Could shape domestic political debate on "
               "Canada's China policy."),
        "zh": "可能影响加拿大国内关于对华政策的政治讨论。",
    },
    "economic": {
        "en": ("May affect economic conditions relevant to Canadian "
               "businesses operating in or with China."),
        "zh": "可能影响与在华或对华经营的加拿大企业相关的经济环境。",
    },
    "social": {
        "en": ("Relevant to diaspora communities, academic exchanges, "
               "or public opinion on Canada-China ties."),
        "zh": "与侨民社区、学术交流或加中关系舆论相关。",
    },
    "legal": {
        "en": ("May affect regulatory frameworks, sanctions compliance, "
               "or rule-of-law considerations."),
        "zh": "可能影响监管框架、制裁合规或法治相关议题。",
    },
}

_WATCH_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "critical": {
        "en": {
            "diplomatic": ("Watch for emergency diplomatic recalls, "
                           "sanctions, or retaliatory measures."),
            "trade": ("Watch for immediate trade disruptions, "
                      "emergency tariffs, or export bans."),
            "military": ("Watch for escalation signals, military "
                         "mobilization, or allied coordination."),
            "technology": ("Watch for technology blacklists, "
                           "emergency export controls, or cyber incidents."),
            "political": ("Watch for parliamentary emergency debates "
                          "or executive policy shifts."),
            "economic": ("Watch for capital flight, currency "
                         "intervention, or investment restrictions."),
            "social": ("Watch for travel advisories, evacuation "
                       "notices, or community safety alerts."),
            "legal": ("Watch for sanctions designations, asset "
                      "freezes, or extradition developments."),
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
            "diplomatic": ("Watch for formal protests, ambassador "
                           "statements, or coalition responses."),
            "trade": ("Watch for new tariff announcements, trade "
                      "investigation launches, or supply chain shifts."),
            "military": ("Watch for military exercises, defence pact "
                         "discussions, or arms sales decisions."),
            "technology": ("Watch for entity list additions, research "
                           "partnership reviews, or data security rules."),
            "political": ("Watch for committee hearings, caucus "
                          "positions, or opposition policy proposals."),
            "economic": ("Watch for investment screening decisions, "
                         "state enterprise activity, or credit actions."),
            "social": ("Watch for university partnership reviews, "
                       "visa policy changes, or diaspora reactions."),
            "legal": ("Watch for new legislation, court rulings, "
                      "or regulatory enforcement actions."),
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
            "trade": "Monitor for trade data releases or business reactions.",
            "military": "Monitor for regional security developments or commentary.",
            "technology": "Monitor for industry responses or regulatory updates.",
            "political": "Monitor for parliamentary questions or media trends.",
            "economic": "Monitor for market reactions or indicator releases.",
            "social": "Monitor for community responses or announcements.",
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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c\u2018\"\'(])', text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 15]


# Filler / transition patterns to penalise
_FILLER_PATTERNS = [
    r"^here (?:are|is) \w+",          # "Here are five ways..."
    r"^(?:but |and |so |yet )",        # conjunction openers
    r"^over the (?:past|last) \w+",    # "Over the past two years..."
    r"^in recent (?:years|months)",
    r"^(?:this|that) (?:comes?|came)",
    r"never been (?:easier|harder)",
]


def _score_sentence(sentence: str, title: str, position: int, total: int) -> float:
    """Score a sentence for informativeness."""
    s_lower = sentence.lower()
    t_lower = title.lower()
    score = 0.0

    # Numbers and data points are the strongest signal of substance
    numbers = re.findall(r'\d+[\d,.]*\s*(?:%|percent|billion|million|thousand|days?|countries)?', sentence)
    score += len(numbers) * 2.0

    # Specific details: proper nouns, quoted speech, named entities
    proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', sentence)
    score += min(len(proper_nouns), 3) * 0.5

    # Title word overlap — sentence explains what headline promises
    title_words = set(re.findall(r'\b\w{4,}\b', t_lower))
    sent_words = set(re.findall(r'\b\w{4,}\b', s_lower))
    overlap = len(title_words & sent_words)
    score += overlap * 1.0

    # Action verbs that indicate substance
    if re.search(r'\b(?:announced?|said|allow|permit|grant|require|impose|launch|sign|ban|approv)', s_lower):
        score += 1.5

    # Penalise filler / transition sentences
    for pat in _FILLER_PATTERNS:
        if re.search(pat, s_lower):
            score -= 3.0
            break

    # Penalise very short sentences
    if len(sentence) < 60:
        score -= 1.0

    # Headings and list items from the fetcher get a boost
    if sentence.startswith("[heading] ") or sentence.startswith("[item] "):
        score += 3.0

    return score


def _extract_list_items(text: str) -> list[str]:
    """Extract [heading] and [item] tagged lines from enriched body text."""
    items: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("[heading] "):
            items.append(line[10:])
        elif line.startswith("[item] "):
            items.append(line[7:])
    return items


def _is_list_headline(title: str) -> bool:
    """Check if headline promises a list (e.g. '5 ways', '3 reasons')."""
    return bool(re.search(r'\b\d+\s+(?:way|reason|thing|tip|step|method|sign|trend|takeaway)', title, re.I))


def _summarize_body(text: str, title: str, max_chars: int = 450) -> str:
    """Produce an extractive summary from article body text.

    For list-style articles (e.g. "5 ways"), extracts headings/items
    and presents them as a concise list.
    For regular articles, picks the most informative sentences by scoring
    on data density, title relevance, and specificity.
    """
    if not text:
        return ""

    # --- List-style articles: extract headings / items ---
    if _is_list_headline(title):
        items = _extract_list_items(text)
        # Only use list format if we found multiple actual list items
        if len(items) >= 2:
            summary_parts: list[str] = []
            total = 0
            for item in items:
                short = item.split(".")[0].strip() if len(item) > 100 else item
                if total + len(short) + 4 > max_chars:
                    break
                summary_parts.append(short)
                total += len(short) + 4
            if len(summary_parts) >= 2:
                return " • ".join(summary_parts)

    # --- Regular articles: extractive summarization ---
    # Remove heading/item lines (standfirsts, subheads) — keep only prose
    lines = [ln for ln in text.split("\n") if not ln.strip().startswith(("[heading]", "[item]"))]
    clean = " ".join(lines)
    sentences = _split_sentences(clean)
    if not sentences:
        return ""

    # Score and rank
    scored: list[tuple[int, float, str]] = []
    for i, sent in enumerate(sentences):
        score = _score_sentence(sent, title, i, len(sentences))
        scored.append((i, score, sent))

    by_score = sorted(scored, key=lambda x: -x[1])

    # Select top sentences within budget
    selected: list[int] = []
    total_len = 0
    for idx, _sc, sent in by_score:
        added = len(sent) + (1 if total_len else 0)
        if total_len > 0 and total_len + added > max_chars:
            continue
        selected.append(idx)
        total_len += added
        if total_len >= max_chars * 0.75:
            break

    if not selected:
        return sentences[0] if sentences else ""

    # Reassemble in original order
    selected.sort()
    return " ".join(sentences[i] for i in selected)


def _normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Normalize a classified signal to conform to the processed schema.

    Converts plain string fields to bilingual format and generates
    rule-based implications from category + severity.
    """
    s = dict(signal)

    # Use full article body (body_text) if available, else RSS snippet
    title_str = s.get("title", "")
    if isinstance(title_str, dict):
        title_str = title_str.get("en", "")
    raw_body = s.pop("body_text", "") or s.pop("body_snippet", "") or s.get("body", "")
    if raw_body and not isinstance(raw_body, dict):
        s["body"] = _summarize_body(raw_body, title_str)

    # Bilingual text fields
    for key in ("title", "body", "source"):
        if key in s:
            s[key] = _to_bilingual(s[key])
        else:
            s[key] = {"en": "", "zh": ""}

    # Normalize date to YYYY-MM-DD
    raw_date = s.get("date", "")
    if raw_date:
        parsed = _parse_signal_date(s)
        if parsed:
            s["date"] = parsed.strftime("%Y-%m-%d")
    else:
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


def _parse_signal_date(signal: dict[str, Any]) -> datetime | None:
    """Try to parse a date from a signal using common formats."""
    raw_date = signal.get("date", "")
    if isinstance(raw_date, dict):
        raw_date = raw_date.get("en", "")
    if not raw_date:
        return None

    # Try common date formats
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%a, %d %b %Y %H:%M:%S %z",  # RSS format
        "%a, %d %b %Y %H:%M:%S %Z",
    ):
        try:
            return datetime.strptime(raw_date[:len(raw_date)], fmt).replace(tzinfo=None)
        except ValueError:
            continue

    # Try ISO-ish prefix: "2026-01-29..."
    m = re.match(r"(\d{4}-\d{2}-\d{2})", raw_date)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass

    return None


# Keywords that indicate Canada-side relevance
_CANADA_KEYWORDS = [
    "canada", "canadian", "ottawa", "trudeau",
    "canola", "huawei", "meng wanzhou",
    "five eyes", "norad", "arctic",
    "bilateral", "canada-china",
]

# Keywords that indicate China-side relevance
_CHINA_KEYWORDS = [
    "china", "chinese", "beijing", "prc",
    "xi jinping", "hong kong", "taiwan",
    "xinjiang", "tibet", "cpc",
]


def _is_bilateral(signal: dict[str, Any]) -> bool:
    """Check if a signal is about Canada-China bilateral relations.

    Requires BOTH Canada-related AND China-related keywords to be present.
    """
    title = signal.get("title", "")
    body = signal.get("body_snippet", signal.get("body", ""))
    if isinstance(title, dict):
        title = title.get("en", "")
    if isinstance(body, dict):
        body = body.get("en", "")
    text = f"{title} {body}".lower()
    has_canada = any(kw in text for kw in _CANADA_KEYWORDS)
    has_china = any(kw in text for kw in _CHINA_KEYWORDS)
    return has_canada and has_china


def _filter_and_prioritize_signals(
    signals: list[dict[str, Any]],
    target_date: str,
    min_signals: int = 10,
    max_signals: int = 25,
) -> list[dict[str, Any]]:
    """Filter signals to recent ones and prioritize bilateral news.

    Uses an adaptive time window: starts at 24 hours and expands
    (48h, 72h, 7d) until at least ``min_signals`` are found.

    Priority order:
      1. Bilateral Canada-China signals (highest priority)
      2. General China / policy signals

    Args:
        signals: All raw signals from fetchers.
        target_date: Pipeline target date (YYYY-MM-DD).
        min_signals: Minimum number of signals before expanding window.
        max_signals: Maximum number of signals to return.

    Returns:
        Filtered and prioritized signal list.
    """
    target_dt = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(hours=23, minutes=59)

    # Pre-parse all signal dates
    dated: list[tuple[dict[str, Any], datetime]] = []
    undated: list[dict[str, Any]] = []
    for signal in signals:
        dt = _parse_signal_date(signal)
        if dt is not None:
            dated.append((signal, dt))
        else:
            undated.append(signal)

    # Adaptive window: expand until we have enough DATED signals.
    # Undated signals (e.g. Xinhua scraped today) are always included
    # but don't count toward the minimum — we want real dated news.
    windows_hours = [24, 48, 72, 168]  # 1d, 2d, 3d, 7d
    recent: list[dict[str, Any]] = []

    for window in windows_hours:
        cutoff = target_dt - timedelta(hours=window)
        recent = [s for s, dt in dated if dt >= cutoff]
        if len(recent) >= min_signals:
            break

    logger.info(
        "Recency filter: %d dated within %dh + %d undated = %d (of %d total)",
        len(recent), window, len(undated), len(recent) + len(undated), len(signals),
    )

    all_candidates = recent + undated

    # Split into bilateral (Canada-China) and general
    bilateral = [s for s in all_candidates if _is_bilateral(s)]
    general = [s for s in all_candidates if not _is_bilateral(s)]

    # Prioritize: bilateral first, then general, up to max_signals
    prioritized = bilateral + general
    return prioritized[:max_signals]


def _transform_market_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform raw yahoo_finance fetcher output to processed schema."""
    indices = []
    for idx in raw.get("indices", []):
        change_pct = idx.get("change_pct", 0)
        direction = "up" if change_pct >= 0 else "down"
        change_str = f"{change_pct:+.2f}%"

        # Convert sparkline array to SVG polyline points
        sparkline = idx.get("sparkline", [])
        sparkline_points = ""
        if sparkline and len(sparkline) >= 2:
            vals = [float(v) for v in sparkline]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1
            pts = []
            for i, v in enumerate(vals):
                x = (i / (len(vals) - 1)) * 100
                y = 32 - ((v - mn) / rng) * 30
                pts.append(f"{x:.0f},{y:.1f}")
            sparkline_points = " ".join(pts)

        indices.append({
            "name": {"en": idx.get("name", ""), "zh": idx.get("name", "")},
            "value": f"{idx.get('value', 0):,.2f}",
            "change": change_str,
            "direction": direction,
            "sparkline_points": sparkline_points,
        })

    # Transform sectors from raw data
    sectors = []
    for sec in raw.get("sectors", []):
        change_pct = sec.get("change_pct", 0)
        direction = "up" if change_pct >= 0 else "down"
        sectors.append({
            "name": {"en": sec.get("name", ""), "zh": sec.get("name", "")},
            "index_name": {"en": sec.get("index_name", sec.get("name", "")),
                           "zh": sec.get("index_name", sec.get("name", ""))},
            "value": f"{sec.get('value', 0):,.2f}" if sec.get("value") else "",
            "change": f"{change_pct:+.2f}%",
            "direction": direction,
        })

    # Transform movers from raw data
    def _fmt_mover(m: dict) -> dict:
        price_val = m.get("close") or m.get("value")
        return {
            "name": {"en": m.get("name", ""), "zh": m.get("name", "")},
            "price": f"HK${price_val:,.2f}" if price_val else "",
            "change": f"{m.get('change_pct', 0):+.2f}%",
        }

    raw_movers = raw.get("movers", {})
    gainers = [_fmt_mover(m) for m in raw_movers.get("gainers", [])]
    losers = [_fmt_mover(m) for m in raw_movers.get("losers", [])]

    return {
        "indices": indices,
        "sectors": sectors,
        "movers": {"gainers": gainers, "losers": losers},
        "ipos": [],
    }


def _transform_trade_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform raw statcan fetcher output to processed schema."""
    imports_m = raw.get("imports_cad_millions", 0)
    exports_m = raw.get("exports_cad_millions", 0)
    balance_m = raw.get("balance_cad_millions", 0)

    def _fmt_cad(val: float) -> dict[str, str]:
        if abs(val) >= 1000:
            return {
                "en": f"${val / 1000:.1f}B CAD",
                "zh": f"{val / 1000:.1f}0亿加元",
            }
        return {
            "en": f"${val:,.0f}M CAD",
            "zh": f"{val:,.0f}百万加元",
        }

    balance_dir = "down" if balance_m < 0 else "up"

    summary_stats = [
        {
            "label": {"en": "Total Imports from China", "zh": "从中国进口总额"},
            "value": _fmt_cad(imports_m),
        },
        {
            "label": {"en": "Total Exports to China", "zh": "对中国出口总额"},
            "value": _fmt_cad(exports_m),
        },
        {
            "label": {"en": "Trade Balance", "zh": "贸易差额"},
            "value": _fmt_cad(balance_m),
            "direction": balance_dir,
        },
    ]

    # Transform commodities into commodity_table for the site template
    commodity_table = []
    for c in raw.get("commodities", []):
        exp_m = c.get("export_cad_millions", 0) or 0
        imp_m = c.get("import_cad_millions", 0) or 0
        bal_m = c.get("balance_cad_millions", exp_m - imp_m)
        trend_val = c.get("trend", "stable")
        disrupted = trend_val.lower() == "disrupted" if isinstance(trend_val, str) else False

        # Map trend to bilingual display
        trend_labels = {
            "up": {"en": "Increasing", "zh": "增长"},
            "down": {"en": "Decreasing", "zh": "下降"},
            "stable": {"en": "Stable", "zh": "稳定"},
            "disrupted": {"en": "Disrupted", "zh": "中断"},
        }
        trend_display = trend_labels.get(
            trend_val.lower() if isinstance(trend_val, str) else "stable",
            {"en": str(trend_val), "zh": str(trend_val)},
        )

        commodity_table.append({
            "commodity": {
                "en": c.get("name", c.get("name_en", "")),
                "zh": c.get("name_zh", c.get("name", "")),
            },
            "export": _fmt_cad(exp_m),
            "import": _fmt_cad(imp_m),
            "balance": _fmt_cad(bal_m),
            "balance_direction": "down" if bal_m < 0 else "up",
            "trend": trend_display,
            "disrupted": disrupted,
        })

    return {
        "summary_stats": summary_stats,
        "commodity_table": commodity_table,
        "totals": raw.get("totals", {}),
        "reference_period": raw.get("reference_period", ""),
    }


def _transform_parliament_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform raw parliament fetcher output to processed schema."""
    # Transform bills to bilingual format
    bills = []
    for b in raw.get("bills", []):
        title_en = b.get("title", "")
        title_zh = b.get("title_fr", title_en)
        status = b.get("status", "")
        # Map status codes to display strings
        status_map = {
            "RoyalAssentGiven": {"en": "Royal Assent", "zh": "御准"},
            "HouseInCommittee": {"en": "In Committee", "zh": "委员会审议中"},
            "HouseAt2ndReading": {
                "en": "2nd Reading", "zh": "二读",
            },
            "SenateInCommittee": {
                "en": "Senate Committee", "zh": "参议院委员会",
            },
        }
        status_display = status_map.get(
            status, {"en": status, "zh": status},
        )
        bills.append({
            "id": b.get("id", ""),
            "title": {"en": title_en, "zh": title_zh},
            "status": status_display,
            "relevance": {"en": "", "zh": ""},
            "last_action": {"en": "", "zh": ""},
        })

    # Transform hansard_stats to hansard
    hs = raw.get("hansard_stats", {})
    total = hs.get("total_mentions", 0)
    by_kw = hs.get("by_keyword", {})

    # Find top keyword
    top_kw = ""
    top_count = 0
    for kw, count in by_kw.items():
        if count > top_count:
            top_kw = kw
            top_count = count

    top_pct = (
        f"{top_count / total * 100:.0f}%" if total > 0 else "0%"
    )
    top_topic = (
        {"en": top_kw, "zh": top_kw} if top_kw
        else {"en": "N/A", "zh": "N/A"}
    )

    hansard = {
        "session_mentions": total,
        "month_mentions": total,
        "top_topic": top_topic,
        "top_topic_pct": top_pct,
    }

    return {"bills": bills, "hansard": hansard}


def _load_supplementary_data(raw_dir: str) -> dict[str, Any]:
    """Load supplementary data (trade, market, parliament) from raw files.

    Transforms raw fetcher output into the processed schema format.

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

    transformers: dict[str, Any] = {
        "trade_data": _transform_trade_data,
        "market_data": _transform_market_data,
        "parliament": _transform_parliament_data,
    }

    file_mapping = {
        "statcan.json": "trade_data",
        "trade.json": "trade_data",
        "yahoo_finance.json": "market_data",
        "market.json": "market_data",
        "parliament.json": "parliament",
    }

    for filename, key in file_mapping.items():
        if result[key] is not None:
            continue
        file_path = raw_path / filename
        if not file_path.exists():
            continue
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
                logger.warning(
                    "Skipping %s (error: %s)",
                    filename, payload["error"],
                )
                continue
            if not isinstance(payload, dict):
                logger.warning("Skipping %s (unexpected format)", filename)
                continue
            # Transform raw data to processed schema
            result[key] = transformers[key](payload)
            logger.info("Loaded and transformed %s", filename)
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

            def _fmt(val: float) -> tuple[str, str]:
                if val >= 1000:
                    return f"${val / 1000:.1f}B", f"{val / 1000:.1f}0亿加元"
                return f"${val:,.0f}M", f"{val:,.0f}百万加元"

            total_en, total_zh = _fmt(total)
            imports_en, imports_zh = _fmt(imports_val)
            exports_en, exports_zh = _fmt(exports_val)

            # Parse reference period for display (e.g. "2025-11-01" → "November 2025")
            ref_period = trade.get("reference_period", "")
            period_en = ref_period
            period_zh = ref_period
            if ref_period and len(ref_period) >= 7:
                try:
                    from datetime import datetime

                    dt = datetime.strptime(ref_period[:7], "%Y-%m")
                    period_en = dt.strftime("%B %Y")
                    month_names_zh = [
                        "", "1月", "2月", "3月", "4月", "5月", "6月",
                        "7月", "8月", "9月", "10月", "11月", "12月",
                    ]
                    period_zh = f"{dt.year}年{month_names_zh[dt.month]}"
                except ValueError:
                    pass

            return {
                "value": {"en": total_en, "zh": total_zh},
                "description": {
                    "en": f"Canada-China bilateral trade ({period_en})",
                    "zh": f"加中双边贸易总额（{period_zh}）",
                },
                "imports": {"en": imports_en, "zh": imports_zh},
                "exports": {"en": exports_en, "zh": exports_zh},
                "reference_period": ref_period,
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


def _extract_market_signals(
    signals: list[dict[str, Any]],
    max_count: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract market signals and regulatory signals from classified signals.

    Market signals: signals with category trade, economic, or technology.
    Regulatory signals: signals with category legal.

    Returns:
        Tuple of (market_signals, regulatory_signals).
    """
    severity_rank = {"critical": 0, "high": 1, "elevated": 2, "moderate": 3, "low": 4}

    market_categories = {"trade", "economic", "technology"}
    regulatory_categories = {"legal"}

    market = []
    regulatory = []

    for s in signals:
        cat = s.get("category", "")
        if cat in market_categories:
            market.append(s)
        elif cat in regulatory_categories:
            regulatory.append(s)

    # Sort by severity, take top N
    market.sort(key=lambda s: severity_rank.get(s.get("severity", "low"), 4))
    regulatory.sort(key=lambda s: severity_rank.get(s.get("severity", "low"), 4))

    return market[:max_count], regulatory[:max_count]


def _generate_quote(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best signal's title as the quote.

    Scoring: prefer bilateral > China-related, higher severity,
    official sources, and more recent dates.
    """
    severity_rank = {"critical": 0, "high": 1, "elevated": 2, "moderate": 3, "low": 4}
    source_rank = {"Global Affairs Canada": 0, "Parliament of Canada": 1, "Xinhua": 2}

    best = None
    best_score = (9, 9, 9, 9)  # (relevance, severity, has_date, source) — lower is better

    for s in signals:
        # Check if the TITLE explicitly mentions China
        title = s.get("title", "")
        if isinstance(title, dict):
            title = title.get("en", "")
        title_lower = title.lower()
        china_in_title = any(
            kw in title_lower
            for kw in ["china", "chinese", "beijing", "xi ", "xi's"]
        )
        bilateral_in_title = china_in_title and any(
            kw in title_lower for kw in ["canada", "canadian", "ottawa"]
        )
        if bilateral_in_title:
            relevance = 0
        elif china_in_title:
            relevance = 1
        else:
            relevance = 2

        sev = severity_rank.get(s.get("severity", "low"), 4)
        src_name = s.get("source", "")
        if isinstance(src_name, dict):
            src_name = src_name.get("en", "")
        src = source_rank.get(src_name, 3)
        # Prefer signals with dates (0) over undated (1)
        has_date = 0 if s.get("date") else 1
        score = (relevance, sev, has_date, src)
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

    # Filter to recent signals and prioritize bilateral news
    raw_signals = _filter_and_prioritize_signals(raw_signals, target_date)

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

    # Step 7c: Extract market & regulatory signals
    market_signals, regulatory_signals = _extract_market_signals(classified_signals)
    logger.info(
        "Extracted %d market signals, %d regulatory signals",
        len(market_signals), len(regulatory_signals),
    )

    # Inject market/regulatory signals into market_data
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


if __name__ == "__main__":
    main()
