"""Data transformation functions for supplementary data and content generation.

Transforms raw fetcher output (market, trade, parliament) into the
processed briefing schema format. Also provides volume numbering,
quote selection, and market signal extraction.

Extracted from cli.py Groups F+G.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("analysis")


def transform_market_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform raw yahoo_finance fetcher output to processed schema."""
    indices = []
    for idx in raw.get("indices", []):
        change_pct = idx.get("change_pct", 0)
        direction = "up" if change_pct >= 0 else "down"
        change_str = f"{change_pct:+.2f}%"

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

    currency_pairs = []
    for pair in raw.get("currency_pairs", []):
        change_pct = pair.get("change_pct", 0) or 0
        direction = "up" if change_pct >= 0 else "down"
        rate = pair.get("rate")
        currency_pairs.append({
            "name": {"en": pair.get("name", ""), "zh": pair.get("name", "")},
            "rate": f"{rate:.4f}" if rate else "",
            "change": f"{change_pct:+.4f}%",
            "direction": direction,
        })

    return {
        "indices": indices,
        "sectors": sectors,
        "movers": {"gainers": gainers, "losers": losers},
        "currency_pairs": currency_pairs,
        "ipos": [],
    }


def transform_trade_data(raw: dict[str, Any]) -> dict[str, Any]:
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

    commodity_table = []
    for c in raw.get("commodities", []):
        exp_m = c.get("export_cad_millions", 0) or 0
        imp_m = c.get("import_cad_millions", 0) or 0
        bal_m = c.get("balance_cad_millions", exp_m - imp_m)
        trend_val = c.get("trend", "stable")
        disrupted = trend_val.lower() == "disrupted" if isinstance(trend_val, str) else False

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


def transform_parliament_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform raw parliament fetcher output to processed schema."""
    bills = []
    for b in raw.get("bills", []):
        title_en = b.get("title", "")
        title_zh = b.get("title_fr", title_en)
        status = b.get("status", "")
        status_map = {
            "RoyalAssentGiven": {"en": "Royal Assent", "zh": "御准"},
            "HouseInCommittee": {"en": "In Committee", "zh": "委员会审议中"},
            "HouseAt2ndReading": {"en": "2nd Reading", "zh": "二读"},
            "SenateInCommittee": {"en": "Senate Committee", "zh": "参议院委员会"},
        }
        status_display = status_map.get(status, {"en": status, "zh": status})
        bills.append({
            "id": b.get("id", ""),
            "title": {"en": title_en, "zh": title_zh},
            "status": status_display,
            "relevance": {"en": "", "zh": ""},
            "last_action": {"en": "", "zh": ""},
        })

    hs = raw.get("hansard_stats", {})
    total = hs.get("total_mentions", 0)
    by_kw = hs.get("by_keyword", {})

    top_kw = ""
    top_count = 0
    for kw, count in by_kw.items():
        if count > top_count:
            top_kw = kw
            top_count = count

    top_pct = f"{top_count / total * 100:.0f}%" if total > 0 else "0%"
    top_topic = {"en": top_kw, "zh": top_kw} if top_kw else {"en": "N/A", "zh": "N/A"}

    hansard = {
        "session_mentions": total,
        "month_mentions": total,
        "top_topic": top_topic,
        "top_topic_pct": top_pct,
    }

    return {"bills": bills, "hansard": hansard}


def load_supplementary_data(raw_dir: str) -> dict[str, Any]:
    """Load supplementary data (trade, market, parliament) from raw files."""
    raw_path = Path(raw_dir)
    result: dict[str, Any] = {
        "trade_data": None,
        "market_data": None,
        "parliament": None,
    }

    transformers: dict[str, Any] = {
        "trade_data": transform_trade_data,
        "market_data": transform_market_data,
        "parliament": transform_parliament_data,
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
            if isinstance(data, dict) and "data" in data:
                payload = data["data"]
            else:
                payload = data
            if isinstance(payload, dict) and "error" in payload:
                logger.warning("Skipping %s (error: %s)", filename, payload["error"])
                continue
            if not isinstance(payload, dict):
                logger.warning("Skipping %s (unexpected format)", filename)
                continue
            result[key] = transformers[key](payload)
            logger.info("Loaded and transformed %s", filename)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", file_path, exc)

    return result


def determine_volume_number(archive_dir: str) -> int:
    """Determine the volume number for today's briefing."""
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


def generate_todays_number(
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

            ref_period = trade.get("reference_period", "")
            period_en = ref_period
            period_zh = ref_period
            if ref_period and len(ref_period) >= 7:
                try:
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

    count = len(signals)
    return {
        "value": {"en": str(count), "zh": str(count)},
        "description": {
            "en": "Canada-China signals tracked today",
            "zh": "今日追踪的加中信号数",
        },
    }


_REGULATORY_KEYWORDS = [
    "regulation", "compliance", "antitrust", "samr", "cac",
    "crackdown", "enforcement", "fine", "penalty", "probe",
    "investigation", "license", "approval",
]


def is_regulatory(
    signal: dict[str, Any],
    regulatory_keywords: list[str] | None = None,
) -> bool:
    """Check if a signal is about regulatory matters."""
    keywords = regulatory_keywords if regulatory_keywords is not None else _REGULATORY_KEYWORDS
    title = signal.get("title", "")
    body = signal.get("body", "")
    if isinstance(title, dict):
        title = title.get("en", "")
    if isinstance(body, dict):
        body = body.get("en", "")
    text = f"{title} {body}".lower()
    return any(kw in text for kw in keywords)


def extract_market_signals(
    signals: list[dict[str, Any]],
    max_count: int = 5,
    regulatory_keywords: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract market signals and regulatory signals from classified signals."""
    severity_rank = {"critical": 0, "high": 1, "elevated": 2, "moderate": 3, "low": 4}
    market_categories = {"trade", "economic", "technology"}

    market = []
    regulatory = []

    for s in signals:
        cat = s.get("category", "")
        if cat in market_categories:
            market.append(s)
        if is_regulatory(s, regulatory_keywords):
            regulatory.append(s)

    market.sort(key=lambda s: severity_rank.get(s.get("severity", "low"), 4))
    regulatory.sort(key=lambda s: severity_rank.get(s.get("severity", "low"), 4))

    return market[:max_count], regulatory[:max_count]


def generate_quote(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best signal's title as the quote."""
    severity_rank = {"critical": 0, "high": 1, "elevated": 2, "moderate": 3, "low": 4}
    source_rank = {"Global Affairs Canada": 0, "Parliament of Canada": 1, "Xinhua": 2}

    best = None
    best_score = (9, 9, 9, 9)

    for s in signals:
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
