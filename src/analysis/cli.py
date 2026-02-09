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
from analysis.dedup import deduplicate_signals, load_recent_signals
from analysis.entities import (
    build_entity_directory,
    match_entities_across_signals,
    match_entities_in_signal,
)
from analysis.llm import llm_generate_perspectives, llm_summarize
from analysis.output import assemble_briefing, validate_briefing, write_archive, write_processed
from analysis.tension_index import compute_tension_index
from analysis.timeline_compiler import (
    compile_canada_china_timeline,
    mark_signal_as_milestone,
    write_timeline,
)
from analysis.translate import (
    _clean_partial_translation,
    translate_to_chinese,
    translate_to_english,
)
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


# ============================================================
# Dual-Perspective Templates (Canada vs Beijing viewpoints)
# ============================================================

# Perspective templates — written for natural readability in both languages
# Canadian perspective: policy implications, stakeholder impacts
# Beijing perspective: based on common state media framing patterns

_CANADA_PERSPECTIVE: dict[str, dict[str, str]] = {
    "diplomatic": {
        "en": ("Ottawa is likely assessing implications for Canada-China "
               "diplomatic engagement. Expect official statements to "
               "balance economic interests with values-based concerns."),
        "zh": ("渥太华正在评估此事对加中外交关系的影响。"
               "预计官方声明将在经济利益与价值观之间寻求平衡。"),
    },
    "trade": {
        "en": ("Canadian exporters should monitor for tariff adjustments "
               "or market access changes. Supply chain diversification "
               "discussions may accelerate."),
        "zh": "加拿大出口商需关注关税调整或市场准入变化。供应链多元化讨论可能加速。",
    },
    "military": {
        "en": ("This intersects with Canada's Indo-Pacific Strategy. "
               "Defence officials and Five Eyes partners will be "
               "tracking developments closely."),
        "zh": "此事与加拿大印太战略相关。国防官员和五眼联盟伙伴将密切追踪事态发展。",
    },
    "technology": {
        "en": ("Universities and research institutions with Chinese "
               "partnerships should review compliance obligations. "
               "Tech sector export controls may be affected."),
        "zh": "与中国有合作的高校和研究机构应审查合规义务。科技行业出口管制可能受影响。",
    },
    "political": {
        "en": ("This will likely draw attention in Parliament. "
               "Cross-party consensus on China policy remains fragile, "
               "so watch for caucus positioning."),
        "zh": "此事可能引发议会关注。各党在对华政策上的共识仍然脆弱，需关注各党团表态。",
    },
    "economic": {
        "en": ("Businesses with China revenue exposure should scenario-plan "
               "for policy volatility. Investment screening rules may "
               "come under review."),
        "zh": "对华业务收入敞口较大的企业应制定政策波动情景预案。投资审查规则可能被重新审视。",
    },
    "social": {
        "en": ("Chinese-Canadian communities may face heightened attention. "
               "Academic institutions should review partnership "
               "arrangements and disclosure requirements."),
        "zh": "华裔加拿大人社区可能面临更多关注。学术机构应审查合作安排和披露要求。",
    },
    "legal": {
        "en": ("Sanctions compliance and export control obligations may "
               "be affected. Legal counsel should monitor for regulatory "
               "guidance updates."),
        "zh": "制裁合规和出口管制义务可能受影响。法律顾问应关注监管指引更新。",
    },
}

_CHINA_PERSPECTIVE: dict[str, dict[str, str]] = {
    "diplomatic": {
        "en": ("Beijing's Foreign Ministry typically frames such matters "
               "as issues of sovereignty and mutual respect. Expect "
               "statements urging 'non-interference in internal affairs.'"),
        "zh": '外交部通常将此类问题定性为主权和相互尊重问题，呼吁"不干涉内政"。',
    },
    "trade": {
        "en": ("State media frames trade disputes as 'protectionism' by "
               "foreign powers. Official line emphasizes China's "
               "commitment to open markets and win-win cooperation."),
        "zh": '官媒将贸易争端定性为外国的"保护主义"。官方立场强调中国坚持开放市场和互利共赢。',
    },
    "military": {
        "en": ("PLA-affiliated media frames regional presence as "
               "'safeguarding territorial integrity.' Foreign military "
               "activities are characterized as 'provocations.'"),
        "zh": '军方媒体将区域军事存在定性为"维护领土完整"。外国军事活动被称为"挑衅行为"。',
    },
    "technology": {
        "en": ("Official policy emphasizes 'technological self-reliance' "
               "(科技自立自强). Export restrictions are framed as "
               "'containment' targeting China's development."),
        "zh": '官方政策强调"科技自立自强"。出口限制被定性为针对中国发展的"遏制"行为。',
    },
    "political": {
        "en": ("State media dismisses foreign criticism as 'interference "
               "in internal affairs.' The Party's governance model is "
               "presented as suited to China's conditions."),
        "zh": '官媒将外国批评斥为"干涉内政"。党的治理模式被阐述为适合中国国情的选择。',
    },
    "economic": {
        "en": ("Economic messaging emphasizes 'high-quality development' "
               "and 'dual circulation.' Policy signals stress stability "
               "while pursuing structural reforms."),
        "zh": '经济宣传强调"高质量发展"和"双循环"。政策信号在推进结构性改革的同时注重稳定。',
    },
    "social": {
        "en": ("Official narratives emphasize 'social harmony' and "
               "'ethnic unity.' Overseas Chinese are positioned as "
               "'bridges' for people-to-people exchanges."),
        "zh": '官方叙事强调"社会和谐"与"民族团结"。海外华人被定位为民间交流的"桥梁"。',
    },
    "legal": {
        "en": ("Legal discourse centers on 'national security' and "
               "'rule of law with Chinese characteristics.' Foreign "
               "legal actions may be characterized as 'long-arm jurisdiction.'"),
        "zh": '法律话语以"国家安全"和"中国特色法治"为核心。外国法律行动可能被定性为"长臂管辖"。',
    },
}

_CHINESE_SOURCE_NAMES: set[str] = {
    # Mainland official
    "xinhua", "新华社", "新华网", "people's daily", "人民日报",
    "global times", "环球时报", "cgtn", "china daily", "中国日报",
    "mfa china", "mofcom", "state council", "国务院", "商务部", "外交部",
    # Mainland business/tech
    "caixin", "财新", "财新网", "the paper", "澎湃", "澎湃新闻",
    "jiemian", "界面", "界面新闻", "36kr", "36氪", "yibang", "亿邦动力",
    # Hong Kong
    "south china morning post", "scmp", "南华早报",
    "scmp diplomacy", "scmp economy", "scmp politics", "scmp china",
    "rthk", "香港電台", "hong kong free press", "hkfp",
    # Taiwan
    "liberty times", "自由時報", "cna", "中央社", "focus taiwan",
    "taipei times", "taiwan news", "united daily news", "聯合報",
    # Diaspora/critical
    "china digital times", "中国数字时代",
    # BBC Chinese
    "bbc china", "bbc中文", "bbc chinese",
}

# Chinese domain patterns for URL-based detection
_CHINESE_DOMAINS: set[str] = {
    "xinhua", "news.cn", "people.com.cn", "globaltimes.cn",
    "chinadaily.com.cn", "cgtn.com", "scmp.com", "thepaper.cn",
    "caixin.com", "jiemian.com", "36kr.com", "yibang.com",
    "cna.com.tw", "focustaiwan.tw", "taipeitimes.com",
    "rthk.hk", "hongkongfp.com",
    "bbc.com/zhongwen", "bbc.co.uk/zhongwen",
}

# Source name translations: Chinese → English
_SOURCE_NAME_TRANSLATIONS: dict[str, str] = {
    # Mainland official
    "人民日报": "People's Daily",
    "新华社": "Xinhua",
    "新华网": "Xinhua",
    "环球时报": "Global Times",
    "中国日报": "China Daily",
    "外交部": "MFA China",
    "商务部": "MOFCOM",
    "国务院": "State Council",
    # Mainland business/tech
    "财新": "Caixin",
    "财新网": "Caixin",
    "澎湃": "The Paper",
    "澎湃新闻": "The Paper",
    "界面": "Jiemian",
    "界面新闻": "Jiemian",
    "36氪": "36Kr",
    "亿邦动力": "Yibang",
    # Hong Kong
    "南华早报": "SCMP",
    "香港電台": "RTHK",
    "香港电台": "RTHK",
    # Taiwan
    "自由時報": "Liberty Times",
    "自由時報國際": "Liberty Times",
    "自由时报": "Liberty Times",
    "中央社": "CNA Taiwan",
    "聯合報": "United Daily News",
    "联合报": "United Daily News",
    # Diaspora
    "中国数字时代": "China Digital Times",
    # BBC
    "BBC中文": "BBC Chinese",
    "BBC中文网": "BBC Chinese",
}


def _translate_source_name(source: str) -> dict[str, str]:
    """Translate a source name to bilingual format.

    If source is in Chinese, looks up translation; otherwise uses as-is.
    """
    if not source:
        return {"en": "", "zh": ""}

    # Check if source is in translation dictionary
    if source in _SOURCE_NAME_TRANSLATIONS:
        return {"en": _SOURCE_NAME_TRANSLATIONS[source], "zh": source}

    # Check partial matches (for variations like "自由時報國際")
    for zh_name, en_name in _SOURCE_NAME_TRANSLATIONS.items():
        if zh_name in source:
            return {"en": en_name, "zh": source}

    # Not a known Chinese source, use as-is
    return {"en": source, "zh": source}


def _is_chinese_source(signal: dict[str, Any]) -> bool:
    """Detect if a signal originates from a Chinese-language source.

    Checks multiple fields: language, region, source name, and URL patterns.
    """
    # Check explicit language/region markers
    if signal.get("language") == "zh":
        return True
    if signal.get("region") in ("mainland", "taiwan", "hongkong"):
        return True

    # Check source name against known Chinese sources
    source = signal.get("source", "")
    if isinstance(source, dict):
        source = f"{source.get('en', '')} {source.get('zh', '')}".lower()
    else:
        source = str(source).lower()
    for known in _CHINESE_SOURCE_NAMES:
        if known in source:
            return True

    # Check URL patterns for Chinese domains
    url = signal.get("url", "") or signal.get("source_url", "") or signal.get("link", "")
    if url:
        url_lower = url.lower()
        for domain in _CHINESE_DOMAINS:
            if domain in url_lower:
                return True

    return False


def _extract_quote(text: str, quote_indicators: list[str]) -> str | None:
    """Try to extract a relevant quote from article text.

    Looks for sentences containing quote indicators like 'said', 'stated',
    'according to', Chinese quote marks, etc.
    """
    if not text:
        return None

    # Split into sentences
    sentences = re.split(r'[.!?。！？]', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30 or len(sentence) > 300:
            continue

        # Check for quote indicators
        for indicator in quote_indicators:
            if indicator in sentence.lower():
                # Clean up the sentence
                clean = re.sub(r'\s+', ' ', sentence).strip()
                if clean:
                    return clean

    return None


def _generate_perspectives(
    category: str,
    is_chinese: bool,
    body_text: str = "",
    source_name: str = "",
    title: str = "",
) -> dict[str, Any]:
    """Generate dual-perspective content for a signal.

    Priority order:
    1. LLM-generated perspectives (signal-specific, when available)
    2. Extracted quotes from article body
    3. Category-based templates (fallback)
    """
    # Try LLM-powered perspectives first for higher-quality, signal-specific content
    if title and body_text:
        llm_perspectives = llm_generate_perspectives(
            title=title,
            body=body_text,
            category=category,
            is_chinese_source=is_chinese,
        )
        if llm_perspectives:
            primary = (
                {"en": "Chinese media", "zh": "中方媒体"} if is_chinese
                else {"en": "Western media", "zh": "西方媒体"}
            )
            return {
                "primary_source": primary,
                "canada": llm_perspectives["canada"],
                "china": llm_perspectives["china"],
                "llm_generated": True,
            }

    # Quote indicators to look for
    en_quote_indicators = [
        "said", "stated", "according to", "told reporters",
        "announced", "emphasized", "warned", "noted",
        "ministry", "spokesman", "official", "government",
    ]
    zh_quote_indicators = [
        "表示", "指出", "强调", "称", "说", "认为",
        "发言人", "外交部", "国务院", "官员",
        '"',  # ASCII double quote
        "\u201c",  # Chinese left double quote "
        "\u300c",  # Chinese corner bracket 「
    ]

    # Try to extract actual quotes
    extracted_quote = None
    if body_text:
        indicators = zh_quote_indicators if is_chinese else en_quote_indicators
        extracted_quote = _extract_quote(body_text, indicators)

    # Get template perspectives as fallback
    canada_template = _CANADA_PERSPECTIVE.get(category, _CANADA_PERSPECTIVE["diplomatic"])
    china_template = _CHINA_PERSPECTIVE.get(category, _CHINA_PERSPECTIVE["diplomatic"])

    # Build perspectives
    primary = (
        {"en": "Chinese media", "zh": "中方媒体"} if is_chinese
        else {"en": "Western media", "zh": "西方媒体"}
    )
    result: dict[str, Any] = {"primary_source": primary}

    # If we have an extracted quote from a Chinese source, use it for Beijing perspective
    if extracted_quote and is_chinese:
        result["china"] = {"en": extracted_quote, "zh": extracted_quote}
        result["china_source"] = {"en": source_name, "zh": source_name}
        result["canada"] = canada_template
    # If we have an extracted quote from Western source, use it for Canada perspective
    elif extracted_quote and not is_chinese:
        result["canada"] = {"en": extracted_quote, "zh": extracted_quote}
        result["canada_source"] = {"en": source_name, "zh": source_name}
        result["china"] = china_template
    # Fall back to templates
    else:
        result["canada"] = canada_template
        result["china"] = china_template

    return result


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
    r"^in \d{4},?\s",                  # "In 1965, ..." historical openings
    r"^(?:back )?in the \d{4}s",       # "In the 1960s..."
    r"^more than (?:the|a) ",          # "More than the chronology..."
    r"^after (?:finishing|completing|graduating)",  # biographical narrative
    r"^(?:he|she|they) (?:was|were) (?:born|raised|assigned)",
    r"^the \d+-year-old",              # "The 24-year-old..."
    r"^(?:one|two|three) of .{0,20}(?:most|first|last)",  # "One of the most loyal..."
]

# Patterns that indicate the sentence is a key point / thesis statement
_KEY_POINT_PATTERNS = [
    r"(?:will|would|may|could|should) (?:continue|remain|face|see|lead|result)",
    r"(?:is|are) (?:expected|likely|set|poised|preparing) to",
    r"(?:announced?|unveiled?|revealed?|confirmed?) (?:that|plans?|a new)",
    r"(?:according to|said|stated|noted|emphasized)",
    r"(?:the|this) (?:move|decision|policy|measure|action) (?:will|would|could|may)",
    r"(?:signals?|indicates?|suggests?|shows?|reflects?) (?:that|a |the )",
    # Geopolitical actors as subject
    r"^(?:china|beijing|the (?:u\.?s\.?|us)|washington|canada|ottawa)",
]


def _score_sentence(sentence: str, title: str, position: int, total: int) -> float:
    """Score a sentence for informativeness."""
    s_lower = sentence.lower()
    t_lower = title.lower()
    score = 0.0

    # Numbers and data points are the strongest signal of substance
    num_pattern = r'\d+[\d,.]*\s*(?:%|percent|billion|million|thousand|days?|countries)?'
    numbers = re.findall(num_pattern, sentence)
    score += len(numbers) * 2.0

    # Specific details: proper nouns, quoted speech, named entities
    proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', sentence)
    score += min(len(proper_nouns), 3) * 0.5

    # Title word overlap — sentence explains what headline promises
    title_words = set(re.findall(r'\b\w{4,}\b', t_lower))
    sent_words = set(re.findall(r'\b\w{4,}\b', s_lower))
    overlap = len(title_words & sent_words)
    score += overlap * 3.0

    # Penalise sentences with no title relevance
    if title_words and overlap == 0:
        score -= 2.0

    # Action verbs that indicate substance
    action_pat = r'\b(?:announced?|said|allow|permit|grant|require|impose|launch|sign|ban|approv)'
    if re.search(action_pat, s_lower):
        score += 1.5

    # Boost sentences that match key point patterns (thesis statements, forward-looking)
    for pat in _KEY_POINT_PATTERNS:
        if re.search(pat, s_lower):
            score += 2.5
            break

    # Penalise filler / transition sentences
    for pat in _FILLER_PATTERNS:
        if re.search(pat, s_lower):
            score -= 4.0
            break

    # Penalise very short sentences
    if len(sentence) < 60:
        score -= 1.0

    # Penalise very long sentences (often narrative/background)
    if len(sentence) > 350:
        score -= 1.5

    # Position bias: first few sentences often contain the lede
    if position < 3:
        score += 1.5
    # Middle of article is often background/history
    elif 0.3 < position / max(total, 1) < 0.7:
        score -= 0.5

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
    list_pat = r'\b\d+\s+(?:way|reason|thing|tip|step|method|sign|trend|takeaway)'
    return bool(re.search(list_pat, title, re.I))


_BOILERPLATE_PATTERNS = [
    # Privacy/cookie notices
    r"(?:our |the )?privacy (?:statement|policy|notice)",
    r"cookie (?:policy|notice|consent)",
    r"by continuing to (?:browse|use|visit)",
    r"you agree to (?:our |the )?use of cookies",
    r"revised privacy policy",
    r"terms of (?:use|service)",
    # Footer/navigation
    r"choose your language",
    r"select (?:your )?language",
    r"subscribe to (?:our )?newsletter",
    r"sign up for (?:our )?newsletter",
    r"follow us on",
    r"share this (?:article|story)",
    # Chinese boilerplate
    r"互联网新闻信息(?:服务)?许可证",
    r"disinformation report hotline",
    r"举报电话",
    r"备案号",
    r"ICP备",
    # Copyright
    r"(?:©|copyright|\(c\))\s*\d{4}",
    r"all rights reserved",
    # Ad/promo
    r"click here to",
    r"read more:",
    r"related (?:articles?|stories?|news):",
]


def _remove_boilerplate(text: str) -> str:
    """Remove website boilerplate text (privacy notices, footers, etc.)."""
    if not text:
        return text

    sentences = _split_sentences(text)
    cleaned = []

    for sent in sentences:
        s_lower = sent.lower()
        is_boilerplate = False
        for pattern in _BOILERPLATE_PATTERNS:
            if re.search(pattern, s_lower, re.IGNORECASE):
                is_boilerplate = True
                break
        if not is_boilerplate:
            cleaned.append(sent)

    return " ".join(cleaned)


def _summarize_body(text: str, title: str, max_chars: int = 500) -> str:
    """Produce an extractive summary from article body text.

    For list-style articles (e.g. "5 ways"), extracts headings/items
    and presents them as a concise list.
    For regular articles, picks the most informative sentences by scoring
    on data density, title relevance, and specificity.
    """
    if not text:
        return ""

    # Remove website boilerplate first
    text = _remove_boilerplate(text)

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

    # Coherence check: if selected sentences are scattered (not from the lede),
    # fall back to the first 2-3 sentences which usually contain the key point
    if selected and selected[0] > 2:
        # None of the selected sentences are from the opening - likely narrative article
        # Fall back to lede sentences
        lede_summary = ""
        for sent in sentences[:3]:
            if len(lede_summary) + len(sent) + 1 <= max_chars:
                lede_summary += (" " if lede_summary else "") + sent
            else:
                break
        if lede_summary:
            return _ensure_complete_sentences(lede_summary)

    summary = " ".join(sentences[i] for i in selected)
    return _ensure_complete_sentences(summary)


def _ensure_complete_sentences(text: str) -> str:
    """Ensure text ends with complete sentences.

    If the text doesn't end with proper punctuation, trim to the last
    complete sentence to avoid truncated summaries.
    """
    if not text:
        return text

    text = text.rstrip()

    # Check if already ends with proper punctuation
    if text[-1] in ".!?。！？":
        return text

    # Find the last punctuation mark
    last_punct = max(
        text.rfind("."),
        text.rfind("!"),
        text.rfind("?"),
        text.rfind("。"),
        text.rfind("！"),
        text.rfind("？"),
    )

    # If we found punctuation and it's not at the very start, trim to it
    if last_punct > 20:  # Require at least 20 chars before punctuation
        return text[: last_punct + 1]

    # No good punctuation found, return original with ellipsis
    return text + "..."


def _normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Normalize a classified signal to conform to the processed schema.

    Converts plain string fields to bilingual format and generates
    rule-based implications from category + severity.
    """
    s = dict(signal)

    # Detect source language
    source_lang = s.get("language", "en")

    # Use full article body (body_text) if available, else RSS snippet
    title_str = s.get("title", "")
    if isinstance(title_str, dict):
        title_str = title_str.get("en", "")
    raw_body = s.pop("body_text", "") or s.pop("body_snippet", "") or s.get("body", "")

    # Preserve original Chinese content before translation (avoids round-trip quality loss)
    preserved_zh_title = None
    preserved_zh_body = None

    # Translate Chinese source to English before summarizing
    if source_lang == "zh" and raw_body and not isinstance(raw_body, dict):
        # Store original Chinese for later use (avoid ZH->EN->ZH round-trip)
        preserved_zh_title = title_str
        preserved_zh_body = raw_body[:2000]  # Keep first 2000 chars of original
        # Translate to English for summarization
        raw_body = translate_to_english([raw_body])[0]
        if title_str:
            translated_title = translate_to_english([title_str])[0]
            # Validate translation succeeded (not still Chinese)
            if _is_primarily_chinese(translated_title):
                # Retry translation
                logger.warning("Title translation failed, retrying: %.50s", title_str)
                translated_title = translate_to_english([title_str])[0]
            # If still Chinese after retry, keep original (will be flagged later)
            if not _is_primarily_chinese(translated_title):
                title_str = translated_title
            else:
                logger.warning("Title translation retry failed: %.50s", title_str)
                # Mark for later handling
                s["_translation_failed"] = True

    # Store preserved content for _translate_signals_batch to use
    if preserved_zh_title:
        s["_preserved_zh_title"] = preserved_zh_title
    if preserved_zh_body:
        s["_preserved_zh_body"] = preserved_zh_body

    if raw_body and not isinstance(raw_body, dict):
        s["body"] = _summarize_body(raw_body, title_str)

        # LLM-enhanced summary for:
        # - critical/high/elevated severity signals
        # - long-form articles (>1500 chars) which are often narrative/analysis
        use_llm = (
            s.get("severity") in ("critical", "high", "elevated")
            or len(raw_body) > 1500
        )
        if use_llm:
            llm_result = llm_summarize(raw_body, title_str)
            if llm_result:
                s["body"] = llm_result

    # Bilingual text fields
    for key in ("title", "body"):
        if key in s:
            s[key] = _to_bilingual(s[key])
        else:
            s[key] = {"en": "", "zh": ""}

    # Source name: use translation dictionary for Chinese sources
    source_val = s.get("source", "")
    if isinstance(source_val, dict):
        s["source"] = source_val
    else:
        s["source"] = _translate_source_name(str(source_val))

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

    # Add dual perspectives (Canadian and Beijing viewpoints)
    # Try to extract actual quotes from article body for more authentic perspectives
    is_chinese = _is_chinese_source(signal)
    body_for_quotes = signal.get("body_text", "") or signal.get("body_snippet", "")
    source_name = signal.get("source", "")
    if isinstance(source_name, dict):
        source_name = source_name.get("en", "") or source_name.get("zh", "")

    # Get title for LLM perspective generation
    title_for_perspectives = s.get("title", {}).get("en", "") or title_str

    s["perspectives"] = _generate_perspectives(
        category=s.get("category", "diplomatic"),
        is_chinese=is_chinese,
        body_text=body_for_quotes,
        source_name=source_name,
        title=title_for_perspectives,
    )
    s["original_zh_source"] = is_chinese

    # Store original source URL for Chinese sources (for "view original" links)
    if is_chinese:
        zh_url = (
            signal.get("url")
            or signal.get("source_url")
            or signal.get("link")
            or signal.get("original_url")
            or ""
        )
        if zh_url:
            s["original_zh_url"] = zh_url

    return s


def _has_english_fragments(text: str, threshold: float = 0.15) -> bool:
    """Check if Chinese text contains significant English fragments."""
    if not text:
        return False
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_chars = sum(1 for c in text if not c.isspace())
    if total_chars == 0:
        return False
    return (ascii_letters / total_chars) > threshold


def _is_primarily_chinese(text: str) -> bool:
    """Check if text is primarily Chinese (CJK characters).

    Used to detect when ZH→EN translation failed and returned Chinese.
    """
    if not text:
        return False
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = sum(1 for c in text if not c.isspace())
    if total_chars == 0:
        return False
    # If more than 30% is CJK, consider it Chinese
    return (cjk_chars / total_chars) > 0.3


def _translate_signals_batch(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate signal titles and bodies to create bilingual content.

    For Chinese-source signals: uses preserved original Chinese (avoids round-trip).
    For English-source signals: translates EN -> ZH.
    For existing signals with bad translations: re-translates to fix quality.

    Truncates bodies to 500 chars before translation.
    Uses LLM primary with MyMemory fallback.
    """
    en_to_zh_texts: list[str] = []
    en_to_zh_map: list[tuple[int, str]] = []  # (signal_idx, field)
    preserved_count = 0
    retranslate_count = 0

    for i, s in enumerate(signals):
        # Check for preserved Chinese content (from Chinese-source signals)
        preserved_title = s.pop("_preserved_zh_title", None)
        preserved_body = s.pop("_preserved_zh_body", None)

        if preserved_title or preserved_body:
            # Use preserved original Chinese instead of round-trip translation
            if preserved_title:
                s["title"]["zh"] = preserved_title
            if preserved_body:
                # Summarize preserved Chinese body to match English summary length
                zh_summary = _summarize_body(preserved_body, preserved_title or "", max_chars=500)
                if zh_summary:
                    s["body"]["zh"] = zh_summary
                else:
                    s["body"]["zh"] = preserved_body[:500]
            preserved_count += 1
            continue

        # Check if existing Chinese translations have quality issues (English fragments)
        # This catches signals from dedup that had bad translations
        title_zh = s.get("title", {}).get("zh", "")
        body_zh = s.get("body", {}).get("zh", "")
        title_en = s.get("title", {}).get("en", "")
        body_en = s.get("body", {}).get("en", "")

        needs_retranslate = False
        if title_zh and _has_english_fragments(title_zh):
            needs_retranslate = True
        if body_zh and _has_english_fragments(body_zh):
            needs_retranslate = True

        if needs_retranslate:
            # Queue for re-translation to fix quality issues
            retranslate_count += 1
            if title_en:
                en_to_zh_texts.append(title_en)
                en_to_zh_map.append((i, "title"))
            if body_en:
                truncated = body_en[:500]
                if len(body_en) > 500:
                    truncated = truncated.rsplit(" ", 1)[0] + "..."
                en_to_zh_texts.append(truncated)
                en_to_zh_map.append((i, "body"))
            continue

        # Skip if already has good Chinese translations (from dedup)
        if title_zh and body_zh:
            continue

        # For English-source signals without Chinese, queue for translation

        if title_en:
            en_to_zh_texts.append(title_en)
            en_to_zh_map.append((i, "title"))
        if body_en:
            truncated = body_en[:500]  # Increased from 300
            if len(body_en) > 500:
                truncated = truncated.rsplit(" ", 1)[0] + "..."
            en_to_zh_texts.append(truncated)
            en_to_zh_map.append((i, "body"))

    if en_to_zh_texts:
        translated = translate_to_chinese(en_to_zh_texts)
        for (sig_idx, field), zh_text in zip(en_to_zh_map, translated):
            signals[sig_idx][field]["zh"] = zh_text

    new_count = len(en_to_zh_texts) - retranslate_count
    logger.info(
        "Translation batch: %d preserved, %d new, %d re-translated for quality",
        preserved_count,
        new_count,
        retranslate_count,
    )

    # Final cleanup pass: fix any remaining English fragments in ALL Chinese text
    # This catches deduped signals and other edge cases
    cleaned_count = 0
    for s in signals:
        for field in ("title", "body"):
            if field in s and isinstance(s[field], dict) and "zh" in s[field]:
                original = s[field]["zh"]
                cleaned = _clean_partial_translation(original)
                if cleaned != original:
                    s[field]["zh"] = cleaned
                    cleaned_count += 1
    if cleaned_count > 0:
        logger.info("Final cleanup fixed %d Chinese text fields", cleaned_count)

    return signals


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

# Broader China-relevance keywords for the analysis pipeline gate.
# A signal must mention at least one of these to be considered relevant.
# Includes both English AND Chinese keywords to catch Chinese-language sources.
_CHINA_RELEVANCE_KEYWORDS = [
    # English keywords
    "china", "chinese", "beijing", "prc", "taiwan", "hong kong",
    "xinjiang", "tibet", "shanghai", "shenzhen", "guangdong",
    "xi jinping", "cpc", "pla", "state council", "npc",
    "huawei", "tiktok", "yuan", "renminbi",
    "south china sea", "one country two systems",
    "canada-china", "sino-canadian", "sino-",
    # Chinese keywords (Simplified)
    "中国", "中华", "北京", "台湾", "香港",
    "新疆", "西藏", "上海", "深圳", "广东",
    "习近平", "国务院", "全国人大", "政协",
    "外交部", "商务部", "人民银行",
    "华为", "人民币", "南海",
    "一带一路", "一国两制",
    "加拿大", "渥太华", "特鲁多",
    # Chinese keywords (Traditional - for Taiwan/HK sources)
    "臺灣", "台灣", "維吾爾", "兩岸",
    "國務院", "習近平", "華為",
    # Common Chinese government/policy terms
    "中共", "党中央", "中央军委", "解放军",
    "发改委", "财政部", "央行",
    # Taiwan-specific keywords (Traditional)
    "軍售", "國防部", "立法院", "民進黨", "國民黨",
    "AIT", "美台", "美方", "中共", "反共",
]

# Keywords indicating LOW-VALUE signals to exclude
# These are signals that mention China geographically but lack policy relevance
_LOW_VALUE_PATTERNS = [
    # Tabloid / accidents / crime without policy angle
    r"\b(?:car accident|traffic accident|car crash|killed in.*(?:crash|accident))\b",
    r"\b(?:crash kills?|dead in|dies? in|death toll)\b",
    r"\b(?:murder|stabbing|assault|robbery|theft|arson)\b",
    r"\b(?:celebrity|gossip|dating|romance|wedding|divorce)\b",
    r"\b(?:sports? (?:star|team)|athlete|tournament|championship|world cup)\b",
    # Pure science without policy implications
    r"\b(?:fossil|dinosaur|archaeological? find|excavation|paleontolog)\b",
    r"\b(?:species discovered|new species|wildlife|biodiversity)\b",
    r"ecological resilience|marsh ecosystem|alpine ecosystem",
    # Entertainment / lifestyle
    r"\b(?:movie|film release|box office|streaming|concert|music video)\b",
    r"\b(?:fashion|beauty|makeup|cosmetic|skincare)\b",
    r"\b(?:food|restaurant|recipe|cuisine|chef)\b",
    # Weather / natural disasters (unless policy-related)
    r"\b(?:earthquake|typhoon|flood|landslide)\b(?!.*(?:policy|aid|relief|government))",
]

# Keywords indicating HIGH-VALUE signals to boost
_HIGH_VALUE_KEYWORDS = [
    # Canada-China bilateral (English)
    "canada-china", "canadian government", "ottawa", "trudeau", "carney",
    "global affairs canada", "parliament", "bill c-",
    # Major China policy (English)
    "xi jinping", "state council", "politburo", "communist party",
    "foreign ministry", "mfa", "ministry of foreign affairs",
    # Geopolitical significance (English)
    "sanctions", "tariff", "trade war", "export ban", "entity list",
    "five eyes", "aukus", "quad", "indo-pacific", "south china sea",
    # Tech/security (English)
    "huawei", "tiktok", "semiconductor", "rare earth", "5g",
    "cyber", "espionage", "interference", "national security",
    # Human rights / values (English)
    "uyghur", "xinjiang", "hong kong", "tibet", "human rights",
    "censorship", "democracy", "crackdown",
    # Chinese keywords - Major policy
    "习近平", "国务院", "中央军委", "政治局", "中共中央",
    "外交部", "商务部", "发改委",
    # Chinese keywords - Geopolitical
    "制裁", "关税", "贸易战", "南海", "台海", "两岸",
    # Chinese keywords - Tech/security
    "华为", "半导体", "芯片", "稀土", "网络安全",
    # Chinese keywords - Canada
    "加拿大", "渥太华", "特鲁多", "加中关系",
]


def _is_china_relevant(signal: dict[str, Any]) -> bool:
    """Check if a signal is relevant to China.

    Returns True if the signal's title or body mentions at least one
    China/bilateral keyword.  This is a broad gate to catch anything
    the fetcher-level filters may have missed.
    """
    title = signal.get("title", "")
    body = signal.get("body_snippet", signal.get("body", ""))
    if isinstance(title, dict):
        title = title.get("en", "")
    if isinstance(body, dict):
        body = body.get("en", "")
    text = f"{title} {body}".lower()
    return any(kw in text for kw in _CHINA_RELEVANCE_KEYWORDS)


def _compute_signal_value(signal: dict[str, Any]) -> tuple[int, str]:
    """Compute a value score for a signal to filter out low-quality content.

    Returns:
        Tuple of (score, reason) where:
        - score >= 2: High value, include
        - score == 1: Medium value, include if space
        - score <= 0: Low value, exclude

    Scoring:
        +3: Direct Canada-China bilateral angle
        +2: Major China policy/political news
        +1: Contains high-value keyword
        -2: Matches low-value pattern (tabloid, pure science, etc.)
    """
    title = signal.get("title", "")
    body = signal.get("body_snippet", signal.get("body", ""))
    if isinstance(title, dict):
        title = title.get("en", "")
    if isinstance(body, dict):
        body = body.get("en", "")

    text = f"{title} {body}".lower()
    title_lower = title.lower()
    score = 0
    reasons = []

    # Check for low-value patterns (tabloid, accidents, pure science)
    for pattern in _LOW_VALUE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score -= 2
            reasons.append(f"low-value pattern: {pattern[:30]}")
            break  # Only penalize once

    # Check for high-value keywords
    high_value_count = 0
    for kw in _HIGH_VALUE_KEYWORDS:
        if kw in text:
            high_value_count += 1

    if high_value_count >= 3:
        score += 2
        reasons.append("multiple high-value keywords")
    elif high_value_count >= 1:
        score += 1
        reasons.append("high-value keyword")

    # Direct bilateral angle (Canada + China in title)
    if any(kw in title_lower for kw in ["canada", "canadian", "ottawa"]):
        if any(kw in title_lower for kw in ["china", "chinese", "beijing"]):
            score += 3
            reasons.append("bilateral in title")

    # Policy signals from official sources get a boost
    source = signal.get("source", "")
    if isinstance(source, dict):
        source = source.get("en", "")
    source_lower = source.lower()
    if any(s in source_lower for s in ["global affairs", "parliament", "xinhua", "mfa", "mofcom"]):
        score += 1
        reasons.append("official source")

    # Canadian media sources get a boost (ensure Canadian perspective in briefing)
    canadian_sources = ["globe and mail", "cbc", "macdonald-laurier", "national post"]
    if any(s in source_lower for s in canadian_sources):
        score += 2
        reasons.append("Canadian source")

    # Chinese-language sources get a boost (ensure Chinese perspective in briefing)
    # This includes mainland, HK, and Taiwan sources
    if any(s in source_lower for s in [
        "人民日报", "新华", "环球时报", "财新", "澎湃", "界面", "36氪",
        "自由時報", "中央社", "香港電台", "南华早报",
        "中国数字时代", "rthk", "scmp",
    ]):
        score += 2
        reasons.append("Chinese source")

    reason_str = "; ".join(reasons) if reasons else "baseline"
    return (score, reason_str)


# Canadian sources we want to ensure appear in briefings
_CANADIAN_SOURCES = {
    "globe and mail", "cbc", "cbc politics", "national post",
    "macdonald-laurier", "global affairs canada", "parliament of canada",
    "canadian press", "toronto star",
}


def _filter_low_value_signals(
    signals: list[dict[str, Any]],
    min_score: int = 0,
) -> list[dict[str, Any]]:
    """Filter out low-value signals based on content analysis.

    Removes signals that are likely tabloid content, random accidents,
    pure science without policy relevance, etc.

    Args:
        signals: List of signals to filter.
        min_score: Minimum value score to keep (default 0).

    Returns:
        Filtered list of signals with value scores attached.
    """
    kept = []
    dropped = []

    for signal in signals:
        score, reason = _compute_signal_value(signal)
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
        for title, score, reason in dropped[:5]:  # Log first 5
            logger.debug("  Dropped: %s (score=%d, %s)", title, score, reason)

    # Source diversity check: warn if no Canadian sources
    _log_source_diversity(kept)

    return kept


def _log_source_diversity(signals: list[dict[str, Any]]) -> None:
    """Log source diversity statistics and warn about missing Canadian sources."""
    from collections import Counter

    sources = []
    canadian_count = 0

    for s in signals:
        source = s.get("source", "")
        if isinstance(source, dict):
            source = source.get("en", "")
        source_lower = source.lower()
        sources.append(source)

        # Check if it's a Canadian source
        if any(cs in source_lower for cs in _CANADIAN_SOURCES):
            canadian_count += 1

    # Count unique sources
    source_counts = Counter(sources)
    unique_sources = len(source_counts)

    logger.info(
        "Source diversity: %d signals from %d unique sources",
        len(signals), unique_sources,
    )

    # Log top sources
    for source, count in source_counts.most_common(5):
        logger.debug("  %s: %d signals", source, count)

    # Warn if no Canadian sources
    if canadian_count == 0 and len(signals) > 0:
        logger.warning(
            "Source diversity warning: No Canadian sources in briefing. "
            "Consider reviewing fetcher RSS feeds or keyword filters."
        )
    else:
        logger.info("Canadian sources: %d signals", canadian_count)


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
    max_signals: int = 75,
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
    windows_hours = [72, 168]  # 3d, 7d — dedup prevents cross-day repeats
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

    # Source-diversified selection: round-robin across sources to ensure
    # no single source dominates the briefing.
    # Bilateral signals get priority, then general — but within each tier,
    # we interleave sources.
    def _source_key(signal: dict[str, Any]) -> str:
        src = signal.get("source", "")
        if isinstance(src, dict):
            src = src.get("en", "")
        # Merge SCMP sub-feeds into "SCMP" for diversity purposes
        if src.startswith("SCMP"):
            return "SCMP"
        return src or "unknown"

    def _round_robin(
        signals: list[dict[str, Any]], max_per_source: int = 3
    ) -> list[dict[str, Any]]:
        from collections import defaultdict
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in signals:
            buckets[_source_key(s)].append(s)
        # Cap each source to max_per_source signals to ensure diversity
        for k in buckets:
            buckets[k] = buckets[k][:max_per_source]
        # Sort source groups: smaller groups first (ensures minority sources
        # get picked before they run out)
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

    diversified = _round_robin(bilateral) + _round_robin(general)
    return diversified[:max_signals]


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

    # Currency pairs
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


_REGULATORY_KEYWORDS = [
    "regulation", "compliance", "antitrust", "samr", "cac",
    "crackdown", "enforcement", "fine", "penalty", "probe",
    "investigation", "license", "approval",
]


def _is_regulatory(signal: dict[str, Any]) -> bool:
    """Check if a signal is about regulatory matters.

    Requires the signal's text to contain at least one regulatory keyword,
    not just a broad "legal" category match.
    """
    title = signal.get("title", "")
    body = signal.get("body", "")
    if isinstance(title, dict):
        title = title.get("en", "")
    if isinstance(body, dict):
        body = body.get("en", "")
    text = f"{title} {body}".lower()
    return any(kw in text for kw in _REGULATORY_KEYWORDS)


def _extract_market_signals(
    signals: list[dict[str, Any]],
    max_count: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract market signals and regulatory signals from classified signals.

    Market signals: signals with category trade, economic, or technology.
    Regulatory signals: signals that contain specific regulatory keywords
    (not just category == "legal", which is too broad).

    Returns:
        Tuple of (market_signals, regulatory_signals).
    """
    severity_rank = {"critical": 0, "high": 1, "elevated": 2, "moderate": 3, "low": 4}

    market_categories = {"trade", "economic", "technology"}

    market = []
    regulatory = []

    for s in signals:
        cat = s.get("category", "")
        if cat in market_categories:
            market.append(s)
        if _is_regulatory(s):
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

    # China-relevance gate: drop signals that don't mention China at all
    pre_count = len(raw_signals)
    raw_signals = [s for s in raw_signals if _is_china_relevant(s)]
    if len(raw_signals) < pre_count:
        logger.info(
            "China-relevance filter: dropped %d of %d signals",
            pre_count - len(raw_signals), pre_count,
        )

    # Value filter: drop low-value signals (tabloid, accidents, pure science)
    pre_count = len(raw_signals)
    raw_signals = _filter_low_value_signals(raw_signals, min_score=0)
    if len(raw_signals) < pre_count:
        logger.info(
            "Value filter: kept %d of %d signals",
            len(raw_signals), pre_count,
        )

    # Pre-classify: add category and entity_ids to raw signals for dedup
    # (Entity-based dedup needs these fields to detect same story from different sources)
    logger.info("Pre-classifying signals for dedup...")
    for signal in raw_signals:
        # Add category if not present
        if "category" not in signal:
            signal["category"] = classify_signal(signal, config.keywords.categories)
        # Add entity_ids if not present
        if "entity_ids" not in signal:
            signal["entity_ids"] = match_entities_in_signal(signal, config.keywords.entity_aliases)

    # Deduplicate signals (within-day + cross-day)
    logger.info("Deduplicating signals...")
    previous_signals = load_recent_signals(
        processed_dir=resolved_output,
        archive_dir=resolved_archive,
        current_date=target_date,
        # lookback_days defaults to 7 in dedup.py
    )
    raw_signals, dedup_stats = deduplicate_signals(raw_signals, previous_signals)

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

    # Step 2b: Translate to Chinese (batch)
    logger.info("Translating signals to Chinese...")
    classified_signals = _translate_signals_batch(classified_signals)

    # Step 2c: Quality filter - remove signals with issues
    pre_count = len(classified_signals)
    quality_filtered = []
    for s in classified_signals:
        # Skip signals with empty bodies (headline-only)
        body_en = s.get("body", {}).get("en", "")
        if not body_en or len(body_en.strip()) < 20:
            title_preview = s.get("title", {}).get("en", "")[:50]
            logger.debug("Dropping signal with empty body: %s", title_preview)
            continue
        # Skip signals where EN title is still Chinese (translation failed)
        title_en = s.get("title", {}).get("en", "")
        if _is_primarily_chinese(title_en):
            logger.warning("Dropping signal with untranslated title: %s", title_en[:50])
            continue
        quality_filtered.append(s)
    classified_signals = quality_filtered
    if len(classified_signals) < pre_count:
        logger.info(
            "Quality filter: dropped %d signals (empty body or translation failure)",
            pre_count - len(classified_signals),
        )

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
    """Compile Canada-China timeline from daily briefings.

    Aggregates daily briefing data into a timeline format suitable for
    visualization. Extracts milestone signals, tension trend data, and
    key events.

    Examples:
        # Compile full timeline from all available briefings
        analysis compile-timeline

        # Compile timeline for specific date range
        analysis compile-timeline --start-date 2025-01-01 --end-date 2025-12-31

        # Use custom directories
        analysis compile-timeline --archive-dir /path/to/archive --timelines-dir /path/to/output
    """
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
    """Mark a signal as a historical milestone.

    Finds a signal by ID in the archive and marks it as a milestone for
    timeline inclusion. Optionally assigns a timeline category.

    Example:
        analysis mark-milestone meng-wanzhou-arrest --timeline-category crisis
    """
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
