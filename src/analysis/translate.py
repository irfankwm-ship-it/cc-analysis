"""Translation module with LLM primary and MyMemory fallback.

Provides translation between English and Simplified Chinese.
Uses local LLM (via ollama) as the primary translator, falling back
to the free MyMemory API when LLM is unavailable.

MyMemory quota: ~5000 chars/day anonymous, higher with email/key.
"""

from __future__ import annotations

import logging
import time

import requests

from analysis.llm import llm_translate

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.mymemory.translated.net/get"


def _mymemory_translate_one(text: str, langpair: str = "en|zh-CN") -> str | None:
    """Translate a single text via MyMemory API.

    Args:
        text: Text to translate.
        langpair: Language pair in MyMemory format (e.g. "en|zh-CN", "zh-CN|en").

    Returns:
        Translated text, or None on failure.
    """
    try:
        resp = requests.get(
            _BASE_URL,
            params={
                "langpair": langpair,
                "q": text,
                "de": "info@chinacompass.ca",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # Check quota
        if data.get("quotaFinished"):
            logger.warning("MyMemory daily quota exhausted")
            return None

        translated = data.get("responseData", {}).get("translatedText", "")
        if translated and translated != text:
            return translated
        return None
    except Exception as exc:
        logger.debug("MyMemory translation failed for (%.40s...): %s", text, exc)
        return None


def _translate_batch(
    texts: list[str],
    source_lang: str,
    target_lang: str,
    mymemory_langpair: str,
) -> list[str]:
    """Translate a list of texts. LLM primary, MyMemory fallback.

    Args:
        texts: Texts to translate.
        source_lang: Source language code ("en" or "zh").
        target_lang: Target language code ("en" or "zh").
        mymemory_langpair: MyMemory langpair format string.

    Returns:
        Translated texts (original text used as fallback on failure).
    """
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not non_empty:
        return texts

    translated = list(texts)  # copy
    llm_success = 0
    mm_success = 0

    for idx, text in non_empty:
        # Try LLM first
        result = llm_translate(text, source_lang, target_lang)
        if result:
            translated[idx] = result
            llm_success += 1
            continue

        # Fall back to MyMemory
        result = _mymemory_translate_one(text, mymemory_langpair)
        if result:
            translated[idx] = result
            mm_success += 1

        # Small delay between MyMemory requests
        time.sleep(0.2)

    total = len(non_empty)
    logger.info(
        "Translation %s→%s: %d/%d via LLM, %d/%d via MyMemory, %d/%d fallback",
        source_lang,
        target_lang,
        llm_success,
        total,
        mm_success,
        total,
        total - llm_success - mm_success,
        total,
    )
    return translated


def translate_to_chinese(texts: list[str]) -> list[str]:
    """Translate a list of English texts to Simplified Chinese.

    LLM primary, MyMemory fallback. Returns original texts for any that fail.
    """
    return _translate_batch(texts, "en", "zh", "en|zh-CN")


def translate_to_english(texts: list[str]) -> list[str]:
    """Translate a list of Chinese texts to English.

    LLM primary, MyMemory fallback. Returns original texts for any that fail.
    """
    return _translate_batch(texts, "zh", "en", "zh-CN|en")
