"""Local LLM client for translation and summarization.

Calls the ollama HTTP API for translation and summarization tasks.
Designed as a best-effort enhancer — returns None on any failure so
callers can fall back to extractive/MyMemory approaches.

Configuration via environment variables:
  OLLAMA_URL  — base URL (default: http://localhost:11434)
  OLLAMA_API_KEY — API key for authenticated proxy (optional)
  OLLAMA_MODEL — model name (default: qwen2.5:3b-instruct-q4_K_M)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))  # Configurable via env var


def _call_ollama(prompt: str) -> str | None:
    """Send a prompt to the ollama API and return the response text.

    Returns None on any failure (network, timeout, parse error).
    """
    url = f"{_OLLAMA_URL}/api/generate"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _OLLAMA_API_KEY:
        headers["X-API-Key"] = _OLLAMA_API_KEY

    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=_TIMEOUT,
            verify=not _OLLAMA_URL.startswith("https://localhost"),
        )
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "").strip()
        if response_text:
            return response_text
        return None
    except Exception as exc:
        logger.warning("Ollama API call failed: %s", exc)
        return None


def llm_translate(text: str, source_lang: str, target_lang: str) -> str | None:
    """Translate text via local LLM.

    Args:
        text: Text to translate.
        source_lang: Source language code (e.g. "zh", "en").
        target_lang: Target language code (e.g. "en", "zh").

    Returns:
        Translated text, or None on failure.
    """
    if not text or not text.strip():
        return None

    lang_names = {"en": "English", "zh": "Simplified Chinese"}
    src_name = lang_names.get(source_lang, source_lang)
    tgt_name = lang_names.get(target_lang, target_lang)

    prompt = (
        f"Translate the following text from {src_name} to {tgt_name}. "
        f"Return ONLY the translation, nothing else.\n\n{text}"
    )

    result = _call_ollama(prompt)
    if result and result != text:
        return result
    return None


def llm_translate_strict(text: str, source_lang: str, target_lang: str) -> str | None:
    """Translate text via local LLM with strict instructions to translate ALL words.

    Use this when the standard translation leaves English words untranslated.

    Args:
        text: Text to translate.
        source_lang: Source language code (e.g. "zh", "en").
        target_lang: Target language code (e.g. "en", "zh").

    Returns:
        Translated text, or None on failure.
    """
    if not text or not text.strip():
        return None

    lang_names = {"en": "English", "zh": "Simplified Chinese"}
    src_name = lang_names.get(source_lang, source_lang)
    tgt_name = lang_names.get(target_lang, target_lang)

    # Build language-specific style guidance
    style_guidance = ""
    if target_lang == "zh":
        style_guidance = (
            "5. Use natural, idiomatic Chinese phrasing — avoid literal translations\n"
            "6. Replace colon-separated phrases (A: B) with natural sentence structures\n"
            "7. For news headlines, use journalistic Chinese style (简洁有力)\n"
            "8. Avoid awkward word-for-word translations; restructure for fluency\n"
        )

    prompt = (
        f"Translate the following text from {src_name} to {tgt_name}.\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. Translate EVERY word including proper nouns, titles, and quoted words\n"
        f"2. Do NOT leave any {src_name} words untranslated\n"
        f"3. For names of people, use standard {tgt_name} transliterations\n"
        f"4. Return ONLY the translation, nothing else\n"
        f"{style_guidance}\n"
        f"Text to translate:\n{text}"
    )

    result = _call_ollama(prompt)
    if result and result != text:
        return result
    return None


def llm_generate_perspectives(
    title: str,
    body: str,
    category: str,
    is_chinese_source: bool,
    lang: str = "en",
) -> dict[str, Any] | None:
    """Generate signal-specific dual perspectives via LLM in a single language.

    Generates Canadian and Beijing viewpoints in the source language using
    plain-text output (no JSON).  The caller is responsible for translating
    to the other language afterwards.

    Args:
        title: Signal headline.
        body: Signal body text (first 1000 chars used).
        category: Signal category (diplomatic, trade, etc.).
        is_chinese_source: Whether the signal originated from a Chinese source.
        lang: Output language — ``"en"`` or ``"zh"``.

    Returns:
        Dict with ``canada`` and ``china`` keys (plain strings in *lang*),
        or None on failure.
    """
    if not title or not body:
        return None

    body_truncated = body[:1000] if len(body) > 1000 else body

    if lang == "zh":
        source_context = "中方媒体" if is_chinese_source else "西方媒体"
        prompt = (
            f"分析以下中加关系新闻，生成两个不同视角的简要评论。\n\n"
            f"类别：{category}\n"
            f"来源：{source_context}\n"
            f"标题：{title}\n"
            f"摘要：{body_truncated}\n\n"
            f"请分别从两个视角各写2-3句话：\n\n"
            f"加拿大视角：[政策影响、利益相关方关切、价值观框架]\n\n"
            f"北京视角：[官方立场、主权关切、官媒叙事]\n\n"
            f"仅输出两个视角内容，不要添加其他文字。"
        )
        marker_canada = "加拿大视角"
        marker_china = "北京视角"
    else:
        source_context = "Chinese media" if is_chinese_source else "Western media"
        prompt = (
            f"Analyze this Canada-China news and generate two perspectives.\n\n"
            f"Category: {category}\n"
            f"Source: {source_context}\n"
            f"Headline: {title}\n"
            f"Summary: {body_truncated}\n\n"
            f"Write exactly two brief perspectives (2-3 sentences each):\n\n"
            f"Canadian perspective: [policy implications, stakeholder concerns, "
            f"values-based framing]\n\n"
            f"Beijing perspective: [official framing, sovereignty concerns, "
            f"state media narrative]\n\n"
            f"Write ONLY the two perspectives, nothing else."
        )
        marker_canada = "canadian perspective"
        marker_china = "beijing perspective"

    result = _call_ollama(prompt)
    if not result:
        return None

    return _parse_perspectives(result, marker_canada, marker_china, lang)


def _parse_perspectives(
    text: str,
    marker_canada: str,
    marker_china: str,
    lang: str,
) -> dict[str, Any] | None:
    """Parse plain-text LLM output into canada/china perspective strings.

    Looks for text markers (case-insensitive) and extracts the text between
    them.  Returns None if either perspective is missing or too short.
    """
    text_lower = text.lower()
    # Also handle variants with colon/fullwidth colon
    ca_idx = -1
    cn_idx = -1
    for variant in (marker_canada, marker_canada + ":", marker_canada + "："):
        pos = text_lower.find(variant.lower())
        if pos >= 0:
            ca_idx = pos + len(variant)
            # Skip colon/space after marker
            while ca_idx < len(text) and text[ca_idx] in ":： \n":
                ca_idx += 1
            break

    for variant in (marker_china, marker_china + ":", marker_china + "："):
        pos = text_lower.find(variant.lower())
        if pos >= 0:
            cn_idx = pos + len(variant)
            while cn_idx < len(text) and text[cn_idx] in ":： \n":
                cn_idx += 1
            break

    if ca_idx < 0 or cn_idx < 0:
        logger.debug("Perspectives markers not found in LLM output")
        return None

    # Extract text between markers
    if ca_idx < cn_idx:
        # Canada comes first — find where China marker label starts
        cn_label_pos = text_lower.find(marker_china.lower())
        canada_text = text[ca_idx:cn_label_pos].strip()
        china_text = text[cn_idx:].strip()
    else:
        # China comes first
        ca_label_pos = text_lower.find(marker_canada.lower())
        china_text = text[cn_idx:ca_label_pos].strip()
        canada_text = text[ca_idx:].strip()

    # Validate minimum length
    min_len = 10 if lang == "zh" else 20
    if len(canada_text) < min_len or len(china_text) < min_len:
        logger.debug(
            "Perspectives too short: canada=%d, china=%d",
            len(canada_text), len(china_text),
        )
        return None

    return {"canada": canada_text, "china": china_text, "lang": lang}


def llm_summarize(
    text: str,
    title: str,
    max_words: int = 100,
    lang: str = "en",
) -> str | None:
    """Summarize article text via local LLM in the specified language.

    Args:
        text: Full article body text.
        title: Article headline for context.
        max_words: Maximum word count (or character count for Chinese).
        lang: Output language — ``"en"`` or ``"zh"``.

    Returns:
        Summary text, or None on failure.
    """
    if not text or not text.strip():
        return None

    # Truncate input to keep prompt reasonable for small model
    truncated = text[:2000]

    if lang == "zh":
        max_chars = max_words * 2  # rough word-to-char ratio
        prompt = (
            f"请将以下文章总结为不超过{max_chars}字。"
            f"聚焦与标题直接相关的事实。"
            f"仅返回摘要内容，不要添加其他文字。\n\n"
            f"标题：{title}\n\n"
            f"文章：{truncated}"
        )
    else:
        prompt = (
            f"Summarize the following article in {max_words} words or fewer. "
            f"Focus on facts directly related to the headline. "
            f"Return ONLY the summary, nothing else.\n\n"
            f"Headline: {title}\n\n"
            f"Article: {truncated}"
        )

    result = _call_ollama(prompt)
    if result:
        # Basic sanity: summary should be shorter than input
        if len(result) < len(text):
            return result
    return None
