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


def llm_summarize(text: str, title: str, max_words: int = 80) -> str | None:
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
