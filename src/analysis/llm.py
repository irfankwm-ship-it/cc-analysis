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
_TIMEOUT = 120


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
) -> dict[str, Any] | None:
    """Generate signal-specific dual perspectives via LLM.

    Creates Canadian and Beijing viewpoints tailored to the specific signal,
    rather than using generic category-based templates.

    Args:
        title: Signal headline.
        body: Signal body text (first 1000 chars used).
        category: Signal category (diplomatic, trade, etc.).
        is_chinese_source: Whether the signal originated from a Chinese source.

    Returns:
        Dict with canada_en, canada_zh, china_en, china_zh keys, or None on failure.
    """
    if not title or not body:
        return None

    # Truncate body for prompt efficiency
    body_truncated = body[:1000] if len(body) > 1000 else body
    source_context = "Chinese media" if is_chinese_source else "Western media"

    prompt = f"""Analyze this Canada-China news and generate dual perspectives.

Category: {category}
Source: {source_context}
Headline: {title}
Summary: {body_truncated}

Generate balanced perspectives from both sides. Each perspective should be 2-3 sentences,
focusing on how each side would view this development and what it means for their interests.

CRITICAL: Respond with ONLY a valid JSON object, no other text:
{{
  "canada_en": "Canadian perspective in English (2-3 sentences)",
  "canada_zh": "加方视角的中文翻译",
  "china_en": "Beijing perspective in English (2-3 sentences)",
  "china_zh": "北京视角的中文翻译"
}}

Canadian perspective: policy implications, stakeholder impacts, values-based concerns.
Beijing perspective: official framing, sovereignty concerns, state media narratives."""

    result = _call_ollama(prompt)
    if not result:
        return None

    # Try to parse JSON from response
    import json

    try:
        # Handle potential markdown code fences
        clean_result = result.strip()
        if clean_result.startswith("```"):
            lines = clean_result.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            clean_result = "\n".join(lines)

        # Find JSON boundaries
        start = clean_result.find("{")
        end = clean_result.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = clean_result[start:end]
            data = json.loads(json_str)

            # Validate required keys
            required = ("canada_en", "canada_zh", "china_en", "china_zh")
            if all(k in data and data[k] for k in required):
                return {
                    "canada": {"en": data["canada_en"], "zh": data["canada_zh"]},
                    "china": {"en": data["china_en"], "zh": data["china_zh"]},
                }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("Failed to parse LLM perspectives response: %s", exc)

    return None


def llm_summarize(text: str, title: str, max_words: int = 100) -> str | None:
    """Summarize article text via local LLM.

    Args:
        text: Full article body text.
        title: Article headline for context.
        max_words: Maximum word count for summary.

    Returns:
        Summary text, or None on failure.
    """
    if not text or not text.strip():
        return None

    # Truncate input to keep prompt reasonable for small model
    truncated = text[:2000]

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
