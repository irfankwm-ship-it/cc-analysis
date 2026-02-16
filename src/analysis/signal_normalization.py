"""Signal normalization: bilingual conversion, implications, perspectives, translation.

Extracted from cli.py Groups C+D. Converts raw/classified signals into
the fully normalized bilingual schema with implications, perspectives,
and translated content.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from opencc import OpenCC

from analysis.llm import llm_generate_perspectives, llm_summarize
from analysis.source_detection import is_chinese_source, translate_source_name
from analysis.text_processing import clean_body_text, summarize_body
from analysis.translate import (
    _clean_partial_translation,
    translate_to_chinese,
    translate_to_english,
)

# Traditional → Simplified Chinese converter (singleton)
_T2S = OpenCC("t2s")

logger = logging.getLogger("analysis")

# Keywords indicating a signal has direct Canada relevance
_CANADA_NEXUS_KEYWORDS = [
    # EN
    "canada", "canadian", "ottawa", "trudeau", "carney", "poilievre",
    "joly", "champagne", "freeland", "csis", "rcmp", "norad",
    "canola", "lobster", "potash", "five eyes", "canada-china",
    "british columbia", "alberta", "ontario", "quebec", "toronto",
    "vancouver", "montreal",
    # ZH
    "加拿大", "渥太华", "特鲁多", "加中", "卡尼", "油菜籽",
]

# Tiered keywords for perspective relevance scoring
_OTTAWA_KEYWORDS: dict[str, list[str]] = {
    # Direct references (weight 5)
    "tier1": [
        "canada", "canadian", "ottawa", "trudeau", "carney", "poilievre",
        "joly", "champagne", "freeland", "csis", "rcmp", "norad",
        "canada-china", "canola", "lobster", "potash",
        "加拿大", "渥太华", "特鲁多", "加中", "卡尼",
    ],
    # Bilateral/alliance context (weight 3)
    "tier2": [
        "five eyes", "nato", "g7", "g20", "aukus", "indo-pacific",
        "bilateral", "allied", "western allies", "sanctions",
        "五眼", "北约", "七国集团",
    ],
    # Broad trade/policy (weight 1)
    "tier3": [
        "tariff", "export control", "import", "supply chain",
        "critical minerals", "arctic", "pacific",
        "关税", "出口管制", "供应链",
    ],
}

_BEIJING_KEYWORDS: dict[str, list[str]] = {
    # Direct China governance/policy (weight 5)
    "tier1": [
        "china", "beijing", "xi jinping", "wang yi", "prc",
        "communist party", "ccp", "state council", "npc",
        "中国", "北京", "习近平", "王毅", "国务院", "中共",
    ],
    # Sovereignty/territorial (weight 4)
    "tier2": [
        "taiwan", "hong kong", "tibet", "xinjiang", "south china sea",
        "one china", "reunification", "sovereignty",
        "台湾", "香港", "西藏", "新疆", "南海", "统一", "主权",
    ],
    # Chinese industry/economy (weight 2)
    "tier3": [
        "huawei", "tencent", "alibaba", "baidu", "bytedance",
        "byd", "catl", "dji", "belt and road", "aiib",
        "华为", "腾讯", "阿里", "百度", "比亚迪", "一带一路",
    ],
}

# Default thresholds for perspective generation gating
_OTTAWA_THRESHOLD = 3
_BEIJING_THRESHOLD = 3


def score_perspective_relevance(title: str, body: str) -> dict[str, int]:
    """Score how relevant a signal is to Ottawa and Beijing perspectives.

    Uses tiered keyword matching: tier1 (direct) = highest weight,
    tier2 (alliance/bilateral) = medium, tier3 (broad) = low.
    Each tier contributes at most once (first match wins per tier).

    Returns dict with ``ottawa`` and ``beijing`` integer scores.
    """
    combined = (title + " " + body).lower()
    combined = _T2S.convert(combined)

    ottawa_score = 0
    for kw in _OTTAWA_KEYWORDS["tier1"]:
        if kw in combined:
            ottawa_score += 5
            break
    for kw in _OTTAWA_KEYWORDS["tier2"]:
        if kw in combined:
            ottawa_score += 3
            break
    for kw in _OTTAWA_KEYWORDS["tier3"]:
        if kw in combined:
            ottawa_score += 1
            break

    beijing_score = 0
    for kw in _BEIJING_KEYWORDS["tier1"]:
        if kw in combined:
            beijing_score += 5
            break
    for kw in _BEIJING_KEYWORDS["tier2"]:
        if kw in combined:
            beijing_score += 4
            break
    for kw in _BEIJING_KEYWORDS["tier3"]:
        if kw in combined:
            beijing_score += 2
            break

    return {"ottawa": ottawa_score, "beijing": beijing_score}


def has_canada_nexus(title: str, body: str) -> bool:
    """Check whether a signal has direct Canada relevance.

    Returns True if any Canada-related keyword appears in the title or body.
    Used to decide whether to generate a Canada-specific perspective
    or a general international-observer perspective.
    """
    combined = (title + " " + body).lower()
    combined = _T2S.convert(combined)
    return any(kw in combined for kw in _CANADA_NEXUS_KEYWORDS)


# Default templates loaded from config; module-level fallbacks for backward compat
_IMPACT_TEMPLATES: dict[str, dict[str, str]] = {}
_WATCH_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {}
_CANADA_PERSPECTIVE: dict[str, dict[str, str]] = {}
_CHINA_PERSPECTIVE: dict[str, dict[str, str]] = {}
_NO_IMPACT_TEMPLATES: dict[str, dict[str, str]] = {}


def _load_default_templates() -> None:
    """Load templates from config YAML (lazy init)."""
    global _IMPACT_TEMPLATES, _WATCH_TEMPLATES, _CANADA_PERSPECTIVE, _CHINA_PERSPECTIVE
    global _NO_IMPACT_TEMPLATES
    if _IMPACT_TEMPLATES:
        return
    try:
        from analysis.config import PROJECT_ROOT, _load_templates
        templates = _load_templates(PROJECT_ROOT / "config")
        _IMPACT_TEMPLATES = templates.impact_templates
        _WATCH_TEMPLATES = templates.watch_templates
        _CANADA_PERSPECTIVE = templates.canada_perspective
        _CHINA_PERSPECTIVE = templates.china_perspective
        _NO_IMPACT_TEMPLATES = templates.no_impact
    except Exception:
        logger.debug("Could not load templates from config; using empty defaults")


def to_bilingual(value: Any) -> dict[str, str]:
    """Ensure a value is in bilingual {"en": ..., "zh": ...} format."""
    if isinstance(value, dict) and "en" in value:
        return value
    text = str(value) if value else ""
    return {"en": text, "zh": text}


def generate_implications(
    category: str,
    severity: str,
    impact_templates: dict[str, dict[str, str]] | None = None,
    watch_templates: dict[str, dict[str, dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Generate rule-based implications from category and severity."""
    _load_default_templates()
    impacts = impact_templates if impact_templates is not None else _IMPACT_TEMPLATES
    watches = watch_templates if watch_templates is not None else _WATCH_TEMPLATES

    impact = impacts.get(category, impacts.get("diplomatic", {"en": "", "zh": ""}))

    severity_key = severity if severity in ("critical", "high") else "default"
    watch_tier = watches.get(severity_key, watches.get("default", {"en": {}, "zh": {}}))
    en_tier = watch_tier.get("en", {})
    zh_tier = watch_tier.get("zh", {})
    watch_en = en_tier.get(category, en_tier.get("diplomatic", ""))
    watch_zh = zh_tier.get(category, zh_tier.get("diplomatic", ""))

    return {
        "canada_impact": impact,
        "what_to_watch": {"en": watch_en, "zh": watch_zh},
    }


def extract_quote(text: str, quote_indicators: list[str]) -> str | None:
    """Try to extract a relevant quote from article text."""
    if not text:
        return None

    sentences = re.split(r'[.!?。！？]', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30 or len(sentence) > 300:
            continue

        for indicator in quote_indicators:
            if indicator in sentence.lower():
                clean = re.sub(r'\s+', ' ', sentence).strip()
                if clean:
                    return clean

    return None


def generate_perspectives(
    category: str,
    is_chinese: bool,
    body_text: str = "",
    source_name: str = "",
    title: str = "",
    lang: str = "en",
    canada_perspective: dict[str, dict[str, str]] | None = None,
    china_perspective: dict[str, dict[str, str]] | None = None,
    ottawa_threshold: int = _OTTAWA_THRESHOLD,
    beijing_threshold: int = _BEIJING_THRESHOLD,
) -> dict[str, Any]:
    """Generate dual-perspective content for a signal in the source language.

    First scores perspective relevance. If Ottawa or Beijing score is below
    threshold, uses a no_impact template instead of generating via LLM.
    Otherwise tries LLM first (single-language plain text), then quote
    extraction, then category templates as fallback. Only populates the
    *lang* side; the other language is filled later by
    ``translate_signals_batch``.
    """
    _load_default_templates()
    ca_persp = canada_perspective if canada_perspective is not None else _CANADA_PERSPECTIVE
    cn_persp = china_perspective if china_perspective is not None else _CHINA_PERSPECTIVE
    no_impact = _NO_IMPACT_TEMPLATES

    primary = (
        {"en": "Chinese media", "zh": "中方媒体"} if is_chinese
        else {"en": "Western media", "zh": "西方媒体"}
    )

    # Score perspective relevance to gate generation
    scores = score_perspective_relevance(title, body_text)
    generate_ottawa = scores["ottawa"] >= ottawa_threshold
    generate_beijing = scores["beijing"] >= beijing_threshold
    logger.debug(
        "Perspective scores for '%s': ottawa=%d, beijing=%d (thresholds: %d/%d)",
        title[:50], scores["ottawa"], scores["beijing"],
        ottawa_threshold, beijing_threshold,
    )

    # Build no_impact fallback dicts
    no_impact_ottawa = no_impact.get("ottawa", {
        "en": "No significant direct impact on Canadian interests identified. "
              "This development is worth monitoring for potential downstream effects.",
        "zh": "目前未发现对加拿大利益的重大直接影响。此事件值得持续关注其潜在的间接效应。",
    })
    no_impact_beijing = no_impact.get("beijing", {
        "en": "No significant implications for Beijing's position identified at this time.",
        "zh": "目前未发现对北京立场的重大影响。",
    })

    # Both below threshold — use no_impact templates for both
    if not generate_ottawa and not generate_beijing:
        return {
            "primary_source": primary,
            "canada": dict(no_impact_ottawa),
            "china": dict(no_impact_beijing),
            "ottawa_score": scores["ottawa"],
            "beijing_score": scores["beijing"],
        }

    # Check Canada relevance to guide perspective generation
    canada_relevant = has_canada_nexus(title, body_text)

    # Determine perspective mode for LLM
    if generate_ottawa and generate_beijing:
        perspective_mode = "both"
    elif generate_beijing:
        perspective_mode = "beijing_only"
    else:
        perspective_mode = "ottawa_only"

    # Try LLM-powered perspectives first (single-language output)
    if title and body_text:
        llm_result = llm_generate_perspectives(
            title=title,
            body=body_text,
            category=category,
            is_chinese_source=is_chinese,
            lang=lang,
            has_canada_nexus=canada_relevant,
            perspective_mode=perspective_mode,
        )
        if llm_result:
            if perspective_mode == "beijing_only":
                # Only Beijing via LLM, Ottawa gets no_impact template
                china_text = _validate_perspective(llm_result["china"], body_text, lang)
                if china_text:
                    if lang == "zh":
                        return {
                            "primary_source": primary,
                            "canada": dict(no_impact_ottawa),
                            "china": {"en": "", "zh": china_text},
                            "llm_generated": True,
                            "ottawa_score": scores["ottawa"],
                            "beijing_score": scores["beijing"],
                        }
                    else:
                        return {
                            "primary_source": primary,
                            "canada": dict(no_impact_ottawa),
                            "china": {"en": china_text, "zh": ""},
                            "llm_generated": True,
                            "ottawa_score": scores["ottawa"],
                            "beijing_score": scores["beijing"],
                        }
            elif perspective_mode == "ottawa_only":
                # Only Ottawa via LLM, Beijing gets no_impact template
                canada_text = _validate_perspective(llm_result["canada"], body_text, lang)
                if canada_text:
                    if lang == "zh":
                        return {
                            "primary_source": primary,
                            "canada": {"en": "", "zh": canada_text},
                            "china": dict(no_impact_beijing),
                            "llm_generated": True,
                            "ottawa_score": scores["ottawa"],
                            "beijing_score": scores["beijing"],
                        }
                    else:
                        return {
                            "primary_source": primary,
                            "canada": {"en": canada_text, "zh": ""},
                            "china": dict(no_impact_beijing),
                            "llm_generated": True,
                            "ottawa_score": scores["ottawa"],
                            "beijing_score": scores["beijing"],
                        }
            else:
                # Both perspectives via LLM (normal path)
                canada_text = _validate_perspective(llm_result["canada"], body_text, lang)
                china_text = _validate_perspective(llm_result["china"], body_text, lang)
                if canada_text and china_text:
                    if lang == "zh":
                        return {
                            "primary_source": primary,
                            "canada": {"en": "", "zh": canada_text},
                            "china": {"en": "", "zh": china_text},
                            "llm_generated": True,
                            "ottawa_score": scores["ottawa"],
                            "beijing_score": scores["beijing"],
                        }
                    else:
                        return {
                            "primary_source": primary,
                            "canada": {"en": canada_text, "zh": ""},
                            "china": {"en": china_text, "zh": ""},
                            "llm_generated": True,
                            "ottawa_score": scores["ottawa"],
                            "beijing_score": scores["beijing"],
                        }
            logger.debug("LLM perspectives failed validation, falling back to template")

    # Quote indicators
    en_quote_indicators = [
        "said", "stated", "according to", "told reporters",
        "announced", "emphasized", "warned", "noted",
        "ministry", "spokesman", "official", "government",
    ]
    zh_quote_indicators = [
        "表示", "指出", "强调", "称", "说", "认为",
        "发言人", "外交部", "国务院", "官员",
        '"', "\u201c", "\u300c",
    ]

    extracted_quote = None
    if body_text:
        indicators = zh_quote_indicators if is_chinese else en_quote_indicators
        extracted_quote = extract_quote(body_text, indicators)

    # Use no_impact template for below-threshold perspective, category template otherwise
    canada_template = (
        dict(no_impact_ottawa) if not generate_ottawa
        else ca_persp.get(category, ca_persp.get("diplomatic", {"en": "", "zh": ""}))
    )
    china_template = (
        dict(no_impact_beijing) if not generate_beijing
        else cn_persp.get(category, cn_persp.get("diplomatic", {"en": "", "zh": ""}))
    )

    result: dict[str, Any] = {
        "primary_source": primary,
        "ottawa_score": scores["ottawa"],
        "beijing_score": scores["beijing"],
    }

    if extracted_quote and is_chinese and generate_beijing:
        result["china"] = {"en": "", "zh": extracted_quote}
        result["china_source"] = {"en": source_name, "zh": source_name}
        result["canada"] = canada_template
    elif extracted_quote and not is_chinese and generate_ottawa:
        result["canada"] = {"en": extracted_quote, "zh": ""}
        result["canada_source"] = {"en": source_name, "zh": source_name}
        result["china"] = china_template
    else:
        result["canada"] = canada_template
        result["china"] = china_template

    return result


def _validate_perspective(text: str, body_text: str, lang: str) -> str | None:
    """Validate a perspective string and return None if it's bad.

    Catches:
    - Empty or too short
    - Article body restatement (>60% word overlap)
    - Language mismatch (CJK in EN field, mostly Latin in ZH field)
    - Scraped content contamination (ads, membership, navigation)
    - Structural markers leaked from prompt
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Too short
    min_len = 15 if lang == "zh" else 30
    if len(text) < min_len:
        return None

    # Scraped content contamination markers
    contamination_markers = [
        "keychain", "shopping bag", "membership", "subscribe",
        "newsletter sign", "HKFP", "follow us", "click here",
        "share this", "read more", "related articles",
        "购物袋", "订阅", "会员", "关注我们",
    ]
    text_lower = text.lower()
    if any(marker.lower() in text_lower for marker in contamination_markers):
        logger.debug("Perspective rejected: scraped content contamination")
        return None

    # Structural markers that shouldn't appear in perspective content
    structural_markers = [
        "OTTAWA:", "BEIJING:", "渥太华：", "北京：",
        "Category:", "Source:", "Title:", "Summary:",
        "类别：", "来源：", "标题：", "摘要：",
    ]
    if any(marker in text for marker in structural_markers):
        logger.debug("Perspective rejected: structural markers found")
        return None

    # Language mismatch: EN field should not be mostly CJK
    if lang == "en":
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total = sum(1 for c in text if not c.isspace())
        if total > 0 and cjk_count / total > 0.3:
            logger.debug("Perspective rejected: CJK in EN field")
            return None

    # Language mismatch: ZH field should have some CJK
    if lang == "zh":
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if cjk_count < 5:
            logger.debug("Perspective rejected: no CJK in ZH field")
            return None

    # Article body restatement detection (high word overlap)
    if body_text and len(body_text) > 50:
        body_lower = body_text[:500].lower()
        # Simple overlap check: count how many 4+ char words from perspective
        # appear verbatim in body
        persp_words = set(re.findall(r'\b\w{4,}\b', text_lower))
        body_words = set(re.findall(r'\b\w{4,}\b', body_lower))
        if persp_words and len(persp_words) > 3:
            overlap = len(persp_words & body_words) / len(persp_words)
            if overlap > 0.8:
                logger.debug("Perspective rejected: body restatement (%.0f%% overlap)", overlap * 100)
                return None

    return text


def _validate_summary(summary: str, title: str, body: str, lang: str) -> bool:
    """Check whether an LLM summary is topically relevant to the source material.

    Computes word/character overlap between the summary and the source
    (title + body). Requires at least 30% of significant words in the
    summary to appear in the source. This catches fabricated summaries
    (e.g. Elon Musk content for an HK tourism article).

    Returns True if the summary is relevant, False otherwise.
    """
    if not summary or not (title or body):
        return True  # nothing to compare against

    source = (title + " " + body).lower()
    summary_lower = summary.lower()

    if lang == "zh":
        # For Chinese, compare 2-char bigrams (pseudo-words)
        source_bigrams = set()
        for i in range(len(source) - 1):
            if '\u4e00' <= source[i] <= '\u9fff' and '\u4e00' <= source[i + 1] <= '\u9fff':
                source_bigrams.add(source[i:i + 2])
        summary_bigrams = set()
        for i in range(len(summary_lower) - 1):
            c1, c2 = summary_lower[i], summary_lower[i + 1]
            if '\u4e00' <= c1 <= '\u9fff' and '\u4e00' <= c2 <= '\u9fff':
                summary_bigrams.add(summary_lower[i:i + 2])
        if len(summary_bigrams) < 3:
            return True  # too few to judge
        overlap = len(summary_bigrams & source_bigrams) / len(summary_bigrams)
    else:
        # For English, compare significant words (4+ chars)
        source_words = set(re.findall(r'\b\w{4,}\b', source))
        summary_words = set(re.findall(r'\b\w{4,}\b', summary_lower))
        if len(summary_words) < 3:
            return True  # too few to judge
        overlap = len(summary_words & source_words) / len(summary_words)

    if overlap < 0.30:
        logger.debug(
            "Summary rejected: %.0f%% overlap (need 30%%): %.60s",
            overlap * 100, summary,
        )
        return False
    return True


def has_english_fragments(text: str, threshold: float = 0.15) -> bool:
    """Check if Chinese text contains significant English fragments."""
    if not text:
        return False
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_chars = sum(1 for c in text if not c.isspace())
    if total_chars == 0:
        return False
    return (ascii_letters / total_chars) > threshold


def is_primarily_chinese(text: str) -> bool:
    """Check if text is primarily Chinese (CJK characters)."""
    if not text:
        return False
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = sum(1 for c in text if not c.isspace())
    if total_chars == 0:
        return False
    return (cjk_chars / total_chars) > 0.3


def normalize_signal(
    signal: dict[str, Any],
    impact_templates: dict[str, dict[str, str]] | None = None,
    watch_templates: dict[str, dict[str, dict[str, str]]] | None = None,
    canada_perspective: dict[str, dict[str, str]] | None = None,
    china_perspective: dict[str, dict[str, str]] | None = None,
    source_names: set[str] | frozenset[str] | None = None,
    domains: set[str] | frozenset[str] | None = None,
    name_translations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize a classified signal to conform to the processed schema.

    Generates summaries and perspectives in the **source language** first.
    The other language is left empty and filled later by
    ``translate_signals_batch``.
    """
    from analysis.signal_filtering import parse_signal_date

    s = dict(signal)

    source_lang = s.get("language", "en")
    is_chinese = is_chinese_source(signal, source_names, domains)
    # Determine effective generation language
    gen_lang = "zh" if (source_lang == "zh" or is_chinese) else "en"

    title_str = s.get("title", "")
    if isinstance(title_str, dict):
        title_str = title_str.get("en", "") or title_str.get("zh", "")
    raw_body = s.pop("body_text", "") or s.pop("body_snippet", "") or s.get("body", "")

    # --- Summarize in source language ---
    if raw_body and not isinstance(raw_body, dict):
        s["body"] = summarize_body(raw_body, title_str)

        use_llm = (
            s.get("severity") in ("critical", "high", "elevated")
            or len(raw_body) > 1500
        )
        if use_llm:
            llm_result = llm_summarize(raw_body, title_str, lang=gen_lang)
            if llm_result and _validate_summary(llm_result, title_str, raw_body, gen_lang):
                s["body"] = llm_result

    # --- Build bilingual shells (populate source-language side only) ---
    for key in ("title", "body"):
        val = s.get(key, "")
        if isinstance(val, dict) and "en" in val:
            continue  # already bilingual
        text = str(val) if val else ""
        if gen_lang == "zh":
            s[key] = {"en": "", "zh": text}
        else:
            s[key] = {"en": text, "zh": ""}

    # --- Source name ---
    source_val = s.get("source", "")
    if isinstance(source_val, dict):
        s["source"] = source_val
    else:
        s["source"] = translate_source_name(str(source_val), name_translations)

    # --- Date ---
    raw_date = s.get("date", "")
    if raw_date:
        parsed = parse_signal_date(s)
        if parsed:
            s["date"] = parsed.strftime("%Y-%m-%d")
    else:
        s["date"] = ""

    # --- Implications (bilingual templates — no change) ---
    if "implications" not in s or not isinstance(s["implications"], dict):
        s["implications"] = generate_implications(
            s.get("category", "diplomatic"),
            s.get("severity", "moderate"),
            impact_templates,
            watch_templates,
        )
    else:
        imp = s["implications"]
        if "canada_impact" not in imp:
            _load_default_templates()
            impacts = impact_templates if impact_templates is not None else _IMPACT_TEMPLATES
            imp["canada_impact"] = impacts.get(
                s.get("category", "diplomatic"),
                impacts.get("diplomatic", {"en": "", "zh": ""}),
            )
        else:
            imp["canada_impact"] = to_bilingual(imp["canada_impact"])
        if "what_to_watch" not in imp or not imp["what_to_watch"]:
            generated = generate_implications(
                s.get("category", "diplomatic"),
                s.get("severity", "moderate"),
                impact_templates,
                watch_templates,
            )
            imp["what_to_watch"] = generated["what_to_watch"]
        else:
            imp["what_to_watch"] = to_bilingual(imp["what_to_watch"])

    # --- Perspectives (generated in source language) ---
    body_for_perspectives = raw_body if raw_body and not isinstance(raw_body, dict) else ""
    if body_for_perspectives:
        body_for_perspectives = clean_body_text(body_for_perspectives)
    source_name_str = signal.get("source", "")
    if isinstance(source_name_str, dict):
        source_name_str = source_name_str.get("en", "") or source_name_str.get("zh", "")

    title_for_perspectives = title_str or s.get("title", {}).get(gen_lang, "")

    s["perspectives"] = generate_perspectives(
        category=s.get("category", "diplomatic"),
        is_chinese=is_chinese,
        body_text=body_for_perspectives,
        source_name=source_name_str,
        title=title_for_perspectives,
        lang=gen_lang,
        canada_perspective=canada_perspective,
        china_perspective=china_perspective,
    )
    s["original_zh_source"] = is_chinese
    s["_source_lang"] = gen_lang  # used by translate_signals_batch

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


def translate_signals_batch(
    signals: list[dict[str, Any]],
    body_truncate_chars: int = 500,
    english_fragment_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """Translate signal titles, bodies, and perspectives to create bilingual content.

    Each signal already has its source-language side populated (title, body,
    perspectives).  This function fills in the missing language:

    - English-source signals: EN already filled → translate to ZH
    - Chinese-source signals: ZH already filled → translate to EN
    """
    en_to_zh_texts: list[str] = []
    en_to_zh_map: list[tuple[int, str, str]] = []  # (idx, field, subfield)
    zh_to_en_texts: list[str] = []
    zh_to_en_map: list[tuple[int, str, str]] = []

    for i, s in enumerate(signals):
        source_lang = s.pop("_source_lang", "en")

        if source_lang == "zh":
            # Chinese source → need English translations
            _collect_missing_translations(
                i, s, "zh", "en", zh_to_en_texts, zh_to_en_map,
                body_truncate_chars,
            )
        else:
            # English source → need Chinese translations
            _collect_missing_translations(
                i, s, "en", "zh", en_to_zh_texts, en_to_zh_map,
                body_truncate_chars,
            )

    # --- Batch translate EN → ZH ---
    if en_to_zh_texts:
        translated = translate_to_chinese(en_to_zh_texts)
        for (sig_idx, field, subfield), text in zip(en_to_zh_map, translated):
            _set_translated(signals[sig_idx], field, subfield, "zh", text)

    # --- Batch translate ZH → EN ---
    if zh_to_en_texts:
        translated = translate_to_english(zh_to_en_texts)
        for (sig_idx, field, subfield), text in zip(zh_to_en_map, translated):
            _set_translated(signals[sig_idx], field, subfield, "en", text)

    logger.info(
        "Translation batch: %d EN→ZH, %d ZH→EN",
        len(en_to_zh_texts), len(zh_to_en_texts),
    )

    # --- Final cleanup of Chinese text ---
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

    # --- Post-translation validation: strip leftover fragments ---
    # Covers title, body, AND perspective fields
    en_cleaned_count = 0
    zh_fallback_count = 0

    def _clean_bilingual_field(
        field: dict[str, str],
        allow_en_fallback: bool = True,
    ) -> None:
        """Strip CJK from EN text; optionally fallback ZH to EN if empty.

        Args:
            field: Bilingual dict with "en" and "zh" keys.
            allow_en_fallback: If True, copy EN→ZH when ZH is empty or
                has no CJK characters. Titles use True (blank title is
                worse than wrong-language title); body and perspectives
                use False (blank is better than English in ZH field).
        """
        nonlocal en_cleaned_count, zh_fallback_count
        en_text = field.get("en", "")
        zh_text = field.get("zh", "")
        if en_text:
            stripped = re.sub(r'[\u4e00-\u9fff]+', '', en_text).strip()
            stripped = re.sub(r'\s{2,}', ' ', stripped)
            if stripped and stripped != en_text:
                field["en"] = stripped
                en_cleaned_count += 1
        if zh_text:
            cjk_count = sum(1 for c in zh_text if '\u4e00' <= c <= '\u9fff')
            if cjk_count == 0:
                if allow_en_fallback:
                    en_fallback = field.get("en", "")
                    if en_fallback:
                        field["zh"] = en_fallback
                        zh_fallback_count += 1
                else:
                    # ZH has no Chinese characters and fallback disabled —
                    # clear it rather than leaving English in a ZH field
                    field["zh"] = ""
        elif not zh_text and field.get("en") and allow_en_fallback:
            field["zh"] = field["en"]
            zh_fallback_count += 1

    for s in signals:
        for key in ("title", "body"):
            field = s.get(key)
            if isinstance(field, dict):
                # Titles: allow EN fallback (blank title worse than EN title)
                # Body: don't allow EN fallback (blank better than wrong language)
                _clean_bilingual_field(field, allow_en_fallback=(key == "title"))
        # Perspectives: don't allow EN fallback (blank better than wrong language)
        persp = s.get("perspectives", {})
        for view in ("canada", "china"):
            view_dict = persp.get(view)
            if isinstance(view_dict, dict):
                _clean_bilingual_field(view_dict, allow_en_fallback=False)

    if en_cleaned_count > 0:
        logger.info("Stripped CJK fragments from %d EN text fields", en_cleaned_count)
    if zh_fallback_count > 0:
        logger.info("Applied EN fallback to %d empty/English-only ZH fields", zh_fallback_count)

    # --- Traditional → Simplified Chinese conversion ---
    t2s_count = 0
    for s in signals:
        for key in ("title", "body"):
            field = s.get(key)
            if isinstance(field, dict) and field.get("zh"):
                converted = _T2S.convert(field["zh"])
                if converted != field["zh"]:
                    field["zh"] = converted
                    t2s_count += 1
        persp = s.get("perspectives", {})
        for view in ("canada", "china"):
            view_dict = persp.get(view)
            if isinstance(view_dict, dict) and view_dict.get("zh"):
                converted = _T2S.convert(view_dict["zh"])
                if converted != view_dict["zh"]:
                    view_dict["zh"] = converted
                    t2s_count += 1
    if t2s_count > 0:
        logger.info("Converted %d Traditional→Simplified Chinese fields", t2s_count)

    return signals


def _collect_missing_translations(
    idx: int,
    signal: dict[str, Any],
    src_lang: str,
    tgt_lang: str,
    texts: list[str],
    mapping: list[tuple[int, str, str]],
    body_truncate_chars: int,
) -> None:
    """Collect texts that need translation from *src_lang* to *tgt_lang*.

    Inspects title, body, and perspective fields, adding any non-empty
    source-language text whose target-language side is empty.
    """
    # Title
    title_src = signal.get("title", {}).get(src_lang, "")
    title_tgt = signal.get("title", {}).get(tgt_lang, "")
    if title_src and not title_tgt:
        texts.append(title_src)
        mapping.append((idx, "title", ""))

    # Body
    body_src = signal.get("body", {}).get(src_lang, "")
    body_tgt = signal.get("body", {}).get(tgt_lang, "")
    if body_src and not body_tgt:
        truncated = body_src[:body_truncate_chars]
        if len(body_src) > body_truncate_chars:
            truncated = truncated.rsplit(" ", 1)[0] + "..." if " " in truncated else truncated
        texts.append(truncated)
        mapping.append((idx, "body", ""))

    # Perspectives — canada and china views
    persp = signal.get("perspectives", {})
    for view in ("canada", "china"):
        view_dict = persp.get(view)
        if not isinstance(view_dict, dict):
            continue
        src_text = view_dict.get(src_lang, "")
        tgt_text = view_dict.get(tgt_lang, "")
        if src_text and not tgt_text:
            texts.append(src_text)
            mapping.append((idx, "perspectives", view))


def _set_translated(
    signal: dict[str, Any],
    field: str,
    subfield: str,
    tgt_lang: str,
    text: str,
) -> None:
    """Set a translated text value on a signal."""
    if field == "perspectives" and subfield:
        persp = signal.get("perspectives", {})
        view_dict = persp.get(subfield)
        if isinstance(view_dict, dict):
            view_dict[tgt_lang] = text
    elif field in ("title", "body"):
        if isinstance(signal.get(field), dict):
            signal[field][tgt_lang] = text
