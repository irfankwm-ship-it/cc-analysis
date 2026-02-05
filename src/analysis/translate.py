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

from analysis.llm import llm_translate, llm_translate_strict

logger = logging.getLogger(__name__)


def _contains_untranslated_english(text: str, threshold: float = 0.15) -> bool:
    """Detect if Chinese text contains significant untranslated English.

    Args:
        text: Text to check (expected to be Chinese).
        threshold: Maximum allowed ratio of ASCII letters to total characters.

    Returns:
        True if text likely contains untranslated English fragments.
    """
    if not text:
        return False
    # Count ASCII letters (a-z, A-Z) - these shouldn't appear in Chinese
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    # Exclude spaces and punctuation from total
    total_chars = sum(1 for c in text if not c.isspace() and c not in ".,;:!?\"'()[]{}—–-")
    if total_chars == 0:
        return False
    ratio = ascii_letters / total_chars
    return ratio > threshold

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

    For English-to-Chinese translations, validates output for untranslated
    English fragments and retries with stricter prompts if detected.

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
    llm_strict_success = 0
    mm_success = 0
    check_english = target_lang == "zh"  # Only check for EN->ZH translations

    for idx, text in non_empty:
        # Try LLM first
        result = llm_translate(text, source_lang, target_lang)
        if result:
            # For Chinese output, check for untranslated English
            if check_english and _contains_untranslated_english(result):
                logger.debug(
                    "LLM translation contains English fragments, retrying strict: %.50s",
                    result,
                )
                # Try strict translation
                strict_result = llm_translate_strict(text, source_lang, target_lang)
                if strict_result and not _contains_untranslated_english(strict_result):
                    translated[idx] = strict_result
                    llm_strict_success += 1
                    continue
                # Try MyMemory as last resort
                mm_result = _mymemory_translate_one(text, mymemory_langpair)
                if mm_result and not _contains_untranslated_english(mm_result):
                    translated[idx] = mm_result
                    mm_success += 1
                    continue
                # Use original LLM result if all else fails
                translated[idx] = result
                llm_success += 1
                continue
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
        "Translation %s->%s: %d/%d LLM, %d/%d LLM-strict, %d/%d MyMemory, %d/%d fallback",
        source_lang,
        target_lang,
        llm_success,
        total,
        llm_strict_success,
        total,
        mm_success,
        total,
        total - llm_success - llm_strict_success - mm_success,
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
