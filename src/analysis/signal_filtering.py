"""Signal loading, filtering, and prioritization.

Extracted from cli.py Group E. Handles raw signal loading from JSON files,
China-relevance gating, value scoring, source diversity, and bilateral
prioritization.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("analysis")

# Default keyword lists — overridable via config params

_CHINA_RELEVANCE_KEYWORDS = [
    "china", "chinese", "beijing", "prc", "taiwan", "hong kong",
    "xinjiang", "tibet", "shanghai", "shenzhen", "guangdong",
    "xi jinping", "cpc", "pla", "state council", "npc",
    "huawei", "tiktok", "yuan", "renminbi",
    "south china sea", "one country two systems",
    "canada-china", "sino-canadian", "sino-",
    "中国", "中华", "北京", "台湾", "香港",
    "新疆", "西藏", "上海", "深圳", "广东",
    "习近平", "国务院", "全国人大", "政协",
    "外交部", "商务部", "人民银行",
    "华为", "人民币", "南海",
    "一带一路", "一国两制",
    "加拿大", "渥太华", "特鲁多",
    "臺灣", "台灣", "維吾爾", "兩岸",
    "國務院", "習近平", "華為",
    "中共", "党中央", "中央军委", "解放军",
    "发改委", "财政部", "央行",
    "軍售", "國防部", "立法院", "民進黨", "國民黨",
    "AIT", "美台", "美方", "中共", "反共",
]

_LOW_VALUE_PATTERNS = [
    r"\b(?:car accident|traffic accident|car crash|killed in.*(?:crash|accident))\b",
    r"\b(?:crash kills?|dead in|dies? in|death toll)\b",
    r"\b(?:murder|stabbing|assault|robbery|theft|arson)\b",
    r"\b(?:celebrity|gossip|dating|romance|wedding|divorce)\b",
    r"\b(?:sports? (?:star|team)|athlete|tournament|championship|world cup)\b",
    r"\b(?:fossil|dinosaur|archaeological? find|excavation|paleontolog)\b",
    r"\b(?:species discovered|new species|wildlife|biodiversity)\b",
    r"ecological resilience|marsh ecosystem|alpine ecosystem",
    r"\b(?:movie|film release|box office|streaming|concert|music video)\b",
    r"\b(?:fashion|beauty|makeup|cosmetic|skincare)\b",
    r"\b(?:food|restaurant|recipe|cuisine|chef)\b",
    r"\b(?:earthquake|typhoon|flood|landslide)\b(?!.*(?:policy|aid|relief|government))",
    # Astronomy / astrophysics
    r"\b(?:black hole|white dwarf|neutron star|supernova|pulsar|quasar)\b",
    r"\b(?:astronomy|astrophysics|cosmology|exoplanet|telescope|observatory)\b",
    r"\b(?:galaxy|galaxies|light-year|stellar|celestial)\b",
    # Sports results / medals (individual sports not caught by existing pattern)
    r"\b(?:skating|medal|luge|bobsled|ski jump|figure skat|speed skat|winter olymp)\b",
    r"\b(?:horse rac|jockey|turf|derby|stakes race)\b",
    # Human interest / viral stories
    r"\b(?:viral|went viral|became famous|internet sensation)\b",
    # Festival / lifestyle
    r"\b(?:new year.{0,10}market|lunar new year.{0,10}fair|spring festival.{0,10}market)\b",
    # Pop culture / fictional characters
    r"\b(?:harry potter|hogwarts|draco malfoy|marvel|avengers|star wars|anime)\b",
    # Chinese low-value patterns (CJK — no word boundaries)
    r"(?:车祸|交通事故|撞车|坠机)",
    r"(?:谋杀|刺伤|袭击|抢劫|盗窃|纵火)",
    r"(?:明星|八卦|绯闻|网红|偶像|选秀)",
    r"(?:体育|运动员|锦标赛|世界杯|联赛|奥运)",
    r"(?:化石|恐龙|考古|古生物)",
    r"(?:黑洞|白矮星|中子星|超新星|天文|望远镜)",
    r"(?:电影|票房|上映|综艺|音乐会|演唱会)",
    r"(?:美妆|化妆品|护肤|时尚|服饰)",
    r"(?:美食|餐厅|食谱|烹饪|菜谱)",
    r"(?:暴雨|降雨|天气预报|气象预警|雷暴)",
    r"(?:楼市|房价|预售|楼盘|房地产(?!.*(?:政策|调控)))",
    # Sports results / medals (Chinese)
    r"(?:滑冰|金牌|奖牌|冬奥|雪橇|短道速滑|花样滑冰)",
    # Horse racing (Traditional + Simplified)
    r"(?:赛马|賽馬|马会|馬會|赛狗)",
    # Viral / trending (human interest)
    r"(?:又火了|走红|爆红|刷屏)",
    # Festival markets / lifestyle
    r"(?:年宵市场|年货|花市|庙会|春节见闻|新春见闻)",
    # Agricultural human interest
    r"(?:菜农|蔬菜.*万斤|水果.*万斤)",
]

_HIGH_VALUE_KEYWORDS = [
    "canada-china", "canadian government", "ottawa", "trudeau", "carney",
    "global affairs canada", "parliament", "bill c-",
    "xi jinping", "state council", "politburo", "communist party",
    "foreign ministry", "mfa", "ministry of foreign affairs",
    "sanctions", "tariff", "trade war", "export ban", "entity list",
    "five eyes", "aukus", "quad", "indo-pacific", "south china sea",
    "huawei", "tiktok", "semiconductor", "rare earth", "5g",
    "cyber", "espionage", "interference", "national security",
    "uyghur", "xinjiang", "hong kong", "tibet", "human rights",
    "censorship", "democracy", "crackdown",
    "习近平", "国务院", "中央军委", "政治局", "中共中央",
    "外交部", "商务部", "发改委",
    "制裁", "关税", "贸易战", "南海", "台海", "两岸",
    "华为", "半导体", "芯片", "稀土", "网络安全",
    "加拿大", "渥太华", "特鲁多", "加中关系",
]

_CANADA_KEYWORDS = [
    "canada", "canadian", "ottawa", "trudeau",
    "canola", "huawei", "meng wanzhou",
    "five eyes", "norad", "arctic",
    "bilateral", "canada-china",
    "加拿大", "渥太华", "特鲁多",
]

_CHINA_KEYWORDS = [
    "china", "chinese", "beijing", "prc",
    "xi jinping", "hong kong", "taiwan",
    "xinjiang", "tibet", "cpc",
    "中国", "中华", "北京", "习近平", "台湾", "香港",
]

_CANADIAN_SOURCES: set[str] = {
    "globe and mail", "cbc", "cbc politics", "national post",
    "macdonald-laurier", "global affairs canada", "parliament of canada",
    "canadian press", "toronto star",
}


def load_raw_signals(raw_dir: str) -> list[dict[str, Any]]:
    """Load raw signal data from the raw directory."""
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

        if isinstance(data, dict) and "data" in data:
            payload = data["data"]
        else:
            payload = data

        if isinstance(payload, list):
            signals.extend(payload)
        elif isinstance(payload, dict):
            for key in ("signals", "articles", "items", "results"):
                if key in payload and isinstance(payload[key], list):
                    signals.extend(payload[key])
                    break
            else:
                if "title" in payload or "headline" in payload:
                    signals.append(payload)

    return signals


def parse_signal_date(signal: dict[str, Any]) -> datetime | None:
    """Try to parse a date from a signal using common formats."""
    raw_date = signal.get("date", "")
    if isinstance(raw_date, dict):
        raw_date = raw_date.get("en", "")
    if not raw_date:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
    ):
        try:
            return datetime.strptime(raw_date[:len(raw_date)], fmt).replace(tzinfo=None)
        except ValueError:
            continue

    m = re.match(r"(\d{4}-\d{2}-\d{2})", raw_date)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass

    return None


def _extract_signal_text(signal: dict[str, Any]) -> tuple[str, str]:
    """Extract combined text and title from a signal in all languages.

    Returns:
        Tuple of (full_text, title_text) — both lowercased.
    """
    title = signal.get("title", "")
    body = signal.get("body_snippet", signal.get("body", ""))

    parts_title: list[str] = []
    parts_body: list[str] = []

    if isinstance(title, dict):
        parts_title.append(title.get("en", ""))
        parts_title.append(title.get("zh", ""))
    elif isinstance(title, str):
        parts_title.append(title)

    if isinstance(body, dict):
        parts_body.append(body.get("en", ""))
        parts_body.append(body.get("zh", ""))
    elif isinstance(body, str):
        parts_body.append(body)

    title_text = " ".join(parts_title).lower()
    full_text = f"{title_text} {' '.join(parts_body)}".lower()
    return full_text, title_text


def is_china_relevant(
    signal: dict[str, Any],
    relevance_keywords: list[str] | None = None,
) -> bool:
    """Check if a signal is relevant to China."""
    keywords = relevance_keywords if relevance_keywords is not None else _CHINA_RELEVANCE_KEYWORDS
    text, _ = _extract_signal_text(signal)
    return any(kw in text for kw in keywords)


def compute_signal_value(
    signal: dict[str, Any],
    high_value_keywords: list[str] | None = None,
    low_value_patterns: list[str] | None = None,
    canadian_sources: set[str] | frozenset[str] | None = None,
) -> tuple[int, str]:
    """Compute a value score for a signal to filter out low-quality content."""
    hv_keywords = high_value_keywords if high_value_keywords is not None else _HIGH_VALUE_KEYWORDS
    lv_patterns = low_value_patterns if low_value_patterns is not None else _LOW_VALUE_PATTERNS
    ca_sources = canadian_sources if canadian_sources is not None else _CANADIAN_SOURCES

    text, title_lower = _extract_signal_text(signal)
    score = 0
    reasons = []

    for pattern in lv_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score -= 2
            reasons.append(f"low-value pattern: {pattern[:30]}")
            break

    high_value_count = 0
    for kw in hv_keywords:
        if kw in text:
            high_value_count += 1

    if high_value_count >= 3:
        score += 2
        reasons.append("multiple high-value keywords")
    elif high_value_count >= 1:
        score += 1
        reasons.append("high-value keyword")

    if any(kw in title_lower for kw in ["canada", "canadian", "ottawa", "加拿大", "渥太华"]):
        if any(kw in title_lower for kw in ["china", "chinese", "beijing", "中国", "北京"]):
            score += 3
            reasons.append("bilateral in title")

    source = signal.get("source", "")
    if isinstance(source, dict):
        source = source.get("en", "")
    source_lower = source.lower()
    if any(s in source_lower for s in ["global affairs", "parliament", "xinhua", "mfa", "mofcom"]):
        score += 1
        reasons.append("official source")

    if any(s in source_lower for s in ca_sources):
        score += 2
        reasons.append("Canadian source")

    if any(s in source_lower for s in [
        "人民日报", "新华", "环球时报", "财新", "澎湃", "界面", "36氪",
        "自由時報", "中央社", "香港電台", "南华早报",
        "中国数字时代", "rthk", "scmp",
    ]):
        score += 2
        reasons.append("Chinese source")

    reason_str = "; ".join(reasons) if reasons else "baseline"
    return (score, reason_str)


def filter_low_value_signals(
    signals: list[dict[str, Any]],
    min_score: int = 0,
    high_value_keywords: list[str] | None = None,
    low_value_patterns: list[str] | None = None,
    canadian_sources: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter out low-value signals based on content analysis."""
    kept = []
    dropped = []

    for signal in signals:
        score, reason = compute_signal_value(
            signal, high_value_keywords, low_value_patterns, canadian_sources
        )
        signal["_value_score"] = score
        signal["_value_reason"] = reason

        if score >= min_score:
            kept.append(signal)
        else:
            title = signal.get("title", "")
            if isinstance(title, dict):
                title = title.get("en", "")
            dropped.append((title[:60], score, reason))

    if dropped:
        logger.info(
            "Value filter: dropped %d low-value signals (min_score=%d)",
            len(dropped), min_score,
        )
        for title, score, reason in dropped[:5]:
            logger.debug("  Dropped: %s (score=%d, %s)", title, score, reason)

    log_source_diversity(kept, canadian_sources)

    return kept


def log_source_diversity(
    signals: list[dict[str, Any]],
    canadian_sources: set[str] | frozenset[str] | None = None,
) -> None:
    """Log source diversity statistics and warn about missing Canadian sources."""
    ca_sources = canadian_sources if canadian_sources is not None else _CANADIAN_SOURCES
    sources = []
    canadian_count = 0

    for s in signals:
        source = s.get("source", "")
        if isinstance(source, dict):
            source = source.get("en", "")
        source_lower = source.lower()
        sources.append(source)

        if any(cs in source_lower for cs in ca_sources):
            canadian_count += 1

    source_counts = Counter(sources)
    unique_sources = len(source_counts)

    logger.info(
        "Source diversity: %d signals from %d unique sources",
        len(signals), unique_sources,
    )

    for source, count in source_counts.most_common(5):
        logger.debug("  %s: %d signals", source, count)

    if canadian_count == 0 and len(signals) > 0:
        logger.warning(
            "Source diversity warning: No Canadian sources in briefing. "
            "Consider reviewing fetcher RSS feeds or keyword filters."
        )
    else:
        logger.info("Canadian sources: %d signals", canadian_count)


def is_bilateral(
    signal: dict[str, Any],
    canada_keywords: list[str] | None = None,
    china_keywords: list[str] | None = None,
) -> bool:
    """Check if a signal is about Canada-China bilateral relations."""
    ca_kw = canada_keywords if canada_keywords is not None else _CANADA_KEYWORDS
    cn_kw = china_keywords if china_keywords is not None else _CHINA_KEYWORDS

    text, _ = _extract_signal_text(signal)
    has_canada = any(kw in text for kw in ca_kw)
    has_china = any(kw in text for kw in cn_kw)
    return has_canada and has_china


def filter_and_prioritize_signals(
    signals: list[dict[str, Any]],
    target_date: str,
    min_signals: int = 10,
    max_signals: int = 75,
    windows_hours: tuple[int, ...] | list[int] = (72, 168),
    max_per_source: int = 3,
    canada_keywords: list[str] | None = None,
    china_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter signals to recent ones and prioritize bilateral news."""
    target_dt = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(hours=23, minutes=59)

    dated: list[tuple[dict[str, Any], datetime]] = []
    undated: list[dict[str, Any]] = []
    for signal in signals:
        dt = parse_signal_date(signal)
        if dt is not None:
            dated.append((signal, dt))
        else:
            undated.append(signal)

    recent: list[dict[str, Any]] = []
    window = windows_hours[-1] if windows_hours else 168

    for w in windows_hours:
        cutoff = target_dt - timedelta(hours=w)
        recent = [s for s, dt in dated if dt >= cutoff]
        window = w
        if len(recent) >= min_signals:
            break

    logger.info(
        "Recency filter: %d dated within %dh + %d undated = %d (of %d total)",
        len(recent), window, len(undated), len(recent) + len(undated), len(signals),
    )

    all_candidates = recent + undated

    bilateral = [s for s in all_candidates if is_bilateral(s, canada_keywords, china_keywords)]
    general = [s for s in all_candidates if not is_bilateral(s, canada_keywords, china_keywords)]

    def _source_key(signal: dict[str, Any]) -> str:
        src = signal.get("source", "")
        if isinstance(src, dict):
            src = src.get("en", "")
        if src.startswith("SCMP"):
            return "SCMP"
        return src or "unknown"

    def _round_robin(
        sigs: list[dict[str, Any]], cap: int = 3
    ) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in sigs:
            buckets[_source_key(s)].append(s)
        for k in buckets:
            buckets[k] = buckets[k][:cap]
        sorted_keys = sorted(buckets.keys(), key=lambda k: len(buckets[k]))
        result: list[dict[str, Any]] = []
        idx = 0
        while True:
            added = False
            for k in sorted_keys:
                if idx < len(buckets[k]):
                    result.append(buckets[k][idx])
                    added = True
            if not added:
                break
            idx += 1
        return result

    diversified = _round_robin(bilateral, max_per_source) + _round_robin(general, max_per_source)
    return diversified[:max_signals]
