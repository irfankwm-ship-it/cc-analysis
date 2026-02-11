"""Chinese source detection and source name translation.

Extracted from cli.py to improve modularity and testability.
"""

from __future__ import annotations

from typing import Any

# Default data — used when no config-loaded data is provided.
# Kept as module-level fallback for backward compatibility.

_CHINESE_SOURCE_NAMES: set[str] = {
    "xinhua", "新华社", "新华网", "people's daily", "人民日报",
    "global times", "环球时报", "cgtn", "china daily", "中国日报",
    "mfa china", "mofcom", "state council", "国务院", "商务部", "外交部",
    "caixin", "财新", "财新网", "the paper", "澎湃", "澎湃新闻",
    "jiemian", "界面", "界面新闻", "36kr", "36氪", "yibang", "亿邦动力",
    "south china morning post", "scmp", "南华早报",
    "scmp diplomacy", "scmp economy", "scmp politics", "scmp china",
    "rthk", "香港電台", "hong kong free press", "hkfp",
    "liberty times", "自由時報", "cna", "中央社", "focus taiwan",
    "taipei times", "taiwan news", "united daily news", "聯合報",
    "china digital times", "中国数字时代",
    "bbc china", "bbc中文", "bbc chinese",
}

_CHINESE_DOMAINS: set[str] = {
    "xinhua", "news.cn", "people.com.cn", "globaltimes.cn",
    "chinadaily.com.cn", "cgtn.com", "scmp.com", "thepaper.cn",
    "caixin.com", "jiemian.com", "36kr.com", "yibang.com",
    "cna.com.tw", "focustaiwan.tw", "taipeitimes.com",
    "rthk.hk", "hongkongfp.com",
    "bbc.com/zhongwen", "bbc.co.uk/zhongwen",
}

_SOURCE_NAME_TRANSLATIONS: dict[str, str] = {
    "人民日报": "People's Daily",
    "新华社": "Xinhua",
    "新华网": "Xinhua",
    "环球时报": "Global Times",
    "中国日报": "China Daily",
    "外交部": "MFA China",
    "商务部": "MOFCOM",
    "国务院": "State Council",
    "财新": "Caixin",
    "财新网": "Caixin",
    "澎湃": "The Paper",
    "澎湃新闻": "The Paper",
    "界面": "Jiemian",
    "界面新闻": "Jiemian",
    "36氪": "36Kr",
    "亿邦动力": "Yibang",
    "南华早报": "SCMP",
    "香港電台": "RTHK",
    "香港电台": "RTHK",
    "自由時報": "Liberty Times",
    "自由時報國際": "Liberty Times",
    "自由时报": "Liberty Times",
    "中央社": "CNA Taiwan",
    "聯合報": "United Daily News",
    "联合报": "United Daily News",
    "中国数字时代": "China Digital Times",
    "BBC中文": "BBC Chinese",
    "BBC中文网": "BBC Chinese",
    "德国之声": "DW Chinese",
    "德國之聲": "DW Chinese",
}


def translate_source_name(
    source: str,
    name_translations: dict[str, str] | None = None,
) -> dict[str, str]:
    """Translate a source name to bilingual format."""
    if not source:
        return {"en": "", "zh": ""}

    translations = name_translations if name_translations is not None else _SOURCE_NAME_TRANSLATIONS

    if source in translations:
        return {"en": translations[source], "zh": source}

    for zh_name, en_name in translations.items():
        if zh_name in source:
            return {"en": en_name, "zh": source}

    return {"en": source, "zh": source}


def is_chinese_source(
    signal: dict[str, Any],
    source_names: set[str] | frozenset[str] | None = None,
    domains: set[str] | frozenset[str] | None = None,
) -> bool:
    """Detect if a signal originates from a Chinese-language source."""
    if signal.get("language") == "zh":
        return True
    if signal.get("region") in ("mainland", "taiwan", "hongkong"):
        return True

    known_sources = source_names if source_names is not None else _CHINESE_SOURCE_NAMES
    known_domains = domains if domains is not None else _CHINESE_DOMAINS

    source = signal.get("source", "")
    if isinstance(source, dict):
        source = f"{source.get('en', '')} {source.get('zh', '')}".lower()
    else:
        source = str(source).lower()
    for known in known_sources:
        if known in source:
            return True

    url = signal.get("url", "") or signal.get("source_url", "") or signal.get("link", "")
    if url:
        url_lower = url.lower()
        for domain in known_domains:
            if domain in url_lower:
                return True

    return False
