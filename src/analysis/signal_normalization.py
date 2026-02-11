"""Signal normalization: bilingual conversion, implications, perspectives, translation.

Extracted from cli.py Groups C+D. Converts raw/classified signals into
the fully normalized bilingual schema with implications, perspectives,
and translated content.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from analysis.llm import llm_generate_perspectives, llm_summarize
from analysis.source_detection import is_chinese_source, translate_source_name
from analysis.text_processing import summarize_body
from analysis.translate import (
    _clean_partial_translation,
    translate_to_chinese,
    translate_to_english,
)

logger = logging.getLogger("analysis")

# Default templates loaded from config; module-level fallbacks for backward compat
_IMPACT_TEMPLATES: dict[str, dict[str, str]] = {}
_WATCH_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {}
_CANADA_PERSPECTIVE: dict[str, dict[str, str]] = {}
_CHINA_PERSPECTIVE: dict[str, dict[str, str]] = {}


def _load_default_templates() -> None:
    """Load templates from config YAML (lazy init)."""
    global _IMPACT_TEMPLATES, _WATCH_TEMPLATES, _CANADA_PERSPECTIVE, _CHINA_PERSPECTIVE
    if _IMPACT_TEMPLATES:
        return
    try:
        from analysis.config import PROJECT_ROOT, _load_templates
        templates = _load_templates(PROJECT_ROOT / "config")
        _IMPACT_TEMPLATES = templates.impact_templates
        _WATCH_TEMPLATES = templates.watch_templates
        _CANADA_PERSPECTIVE = templates.canada_perspective
        _CHINA_PERSPECTIVE = templates.china_perspective
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
    canada_perspective: dict[str, dict[str, str]] | None = None,
    china_perspective: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate dual-perspective content for a signal."""
    _load_default_templates()
    ca_persp = canada_perspective if canada_perspective is not None else _CANADA_PERSPECTIVE
    cn_persp = china_perspective if china_perspective is not None else _CHINA_PERSPECTIVE

    # Try LLM-powered perspectives first
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

    canada_template = ca_persp.get(category, ca_persp.get("diplomatic", {"en": "", "zh": ""}))
    china_template = cn_persp.get(category, cn_persp.get("diplomatic", {"en": "", "zh": ""}))

    primary = (
        {"en": "Chinese media", "zh": "中方媒体"} if is_chinese
        else {"en": "Western media", "zh": "西方媒体"}
    )
    result: dict[str, Any] = {"primary_source": primary}

    if extracted_quote and is_chinese:
        result["china"] = {"en": extracted_quote, "zh": extracted_quote}
        result["china_source"] = {"en": source_name, "zh": source_name}
        result["canada"] = canada_template
    elif extracted_quote and not is_chinese:
        result["canada"] = {"en": extracted_quote, "zh": extracted_quote}
        result["canada_source"] = {"en": source_name, "zh": source_name}
        result["china"] = china_template
    else:
        result["canada"] = canada_template
        result["china"] = china_template

    return result


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
    """Normalize a classified signal to conform to the processed schema."""
    from analysis.signal_filtering import parse_signal_date

    s = dict(signal)

    source_lang = s.get("language", "en")

    title_str = s.get("title", "")
    if isinstance(title_str, dict):
        title_str = title_str.get("en", "")
    raw_body = s.pop("body_text", "") or s.pop("body_snippet", "") or s.get("body", "")

    preserved_zh_title = None
    preserved_zh_body = None

    if source_lang == "zh" and raw_body and not isinstance(raw_body, dict):
        preserved_zh_title = title_str
        preserved_zh_body = raw_body[:2000]
        raw_body = translate_to_english([raw_body])[0]
        if title_str:
            translated_title = translate_to_english([title_str])[0]
            if is_primarily_chinese(translated_title):
                logger.warning("Title translation failed, retrying: %.50s", title_str)
                translated_title = translate_to_english([title_str])[0]
            if not is_primarily_chinese(translated_title):
                title_str = translated_title
            else:
                logger.warning("Title translation retry failed: %.50s", title_str)
                s["_translation_failed"] = True

    if preserved_zh_title:
        s["_preserved_zh_title"] = preserved_zh_title
    if preserved_zh_body:
        s["_preserved_zh_body"] = preserved_zh_body

    if raw_body and not isinstance(raw_body, dict):
        s["body"] = summarize_body(raw_body, title_str)

        use_llm = (
            s.get("severity") in ("critical", "high", "elevated")
            or len(raw_body) > 1500
        )
        if use_llm:
            llm_result = llm_summarize(raw_body, title_str)
            if llm_result:
                s["body"] = llm_result

    for key in ("title", "body"):
        if key in s:
            s[key] = to_bilingual(s[key])
        else:
            s[key] = {"en": "", "zh": ""}

    source_val = s.get("source", "")
    if isinstance(source_val, dict):
        s["source"] = source_val
    else:
        s["source"] = translate_source_name(str(source_val), name_translations)

    raw_date = s.get("date", "")
    if raw_date:
        parsed = parse_signal_date(s)
        if parsed:
            s["date"] = parsed.strftime("%Y-%m-%d")
    else:
        s["date"] = ""

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

    is_chinese = is_chinese_source(signal, source_names, domains)
    body_for_quotes = signal.get("body_text", "") or signal.get("body_snippet", "")
    source_name_str = signal.get("source", "")
    if isinstance(source_name_str, dict):
        source_name_str = source_name_str.get("en", "") or source_name_str.get("zh", "")

    title_for_perspectives = s.get("title", {}).get("en", "") or title_str

    s["perspectives"] = generate_perspectives(
        category=s.get("category", "diplomatic"),
        is_chinese=is_chinese,
        body_text=body_for_quotes,
        source_name=source_name_str,
        title=title_for_perspectives,
        canada_perspective=canada_perspective,
        china_perspective=china_perspective,
    )
    s["original_zh_source"] = is_chinese

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
    """Translate signal titles and bodies to create bilingual content."""
    en_to_zh_texts: list[str] = []
    en_to_zh_map: list[tuple[int, str]] = []
    preserved_count = 0
    retranslate_count = 0

    for i, s in enumerate(signals):
        preserved_title = s.pop("_preserved_zh_title", None)
        preserved_body = s.pop("_preserved_zh_body", None)

        if preserved_title or preserved_body:
            if preserved_title:
                s["title"]["zh"] = preserved_title
            if preserved_body:
                zh_summary = summarize_body(
                    preserved_body, preserved_title or "", max_chars=body_truncate_chars
                )
                if zh_summary:
                    s["body"]["zh"] = zh_summary
                else:
                    s["body"]["zh"] = preserved_body[:body_truncate_chars]
            preserved_count += 1
            continue

        title_zh = s.get("title", {}).get("zh", "")
        body_zh = s.get("body", {}).get("zh", "")
        title_en = s.get("title", {}).get("en", "")
        body_en = s.get("body", {}).get("en", "")

        needs_retranslate = False
        if title_zh and has_english_fragments(title_zh, english_fragment_threshold):
            needs_retranslate = True
        if body_zh and has_english_fragments(body_zh, english_fragment_threshold):
            needs_retranslate = True

        if needs_retranslate:
            retranslate_count += 1
            if title_en:
                en_to_zh_texts.append(title_en)
                en_to_zh_map.append((i, "title"))
            if body_en:
                truncated = body_en[:body_truncate_chars]
                if len(body_en) > body_truncate_chars:
                    truncated = truncated.rsplit(" ", 1)[0] + "..."
                en_to_zh_texts.append(truncated)
                en_to_zh_map.append((i, "body"))
            continue

        if title_zh and body_zh:
            continue

        if title_en:
            en_to_zh_texts.append(title_en)
            en_to_zh_map.append((i, "title"))
        if body_en:
            truncated = body_en[:body_truncate_chars]
            if len(body_en) > body_truncate_chars:
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
