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


# Common English words that LLMs often leave untranslated
_UNTRANSLATED_WORDS = {
    "capitulation": "投降",
    "imperative": "当务之急",
    "unacceptable": "不可接受",
    "vibrant": "充满活力",
    "likely": "可能",
    "unlikely": "不太可能",
    "aggressive": "激进",
    "contingency": "应急",
    "intelligence-sharing": "情报共享",
    "expansion": "扩张",
    "comprehensive": "全面",
    "significant": "重大",
    "unprecedented": "史无前例",
    "strategic": "战略性",
    "bilateral": "双边",
    "multilateral": "多边",
    "escalation": "升级",
    "de-escalation": "缓和",
    "retaliation": "报复",
    "sanctions": "制裁",
    "tariffs": "关税",
    "decoupling": "脱钩",
    # Additional commonly untranslated words
    "diet": "国会",  # Japanese Diet
    "visiting": "访问",
    "oman": "阿曼",
    "assisted reproductive": "辅助生殖",
    "marines": "海军陆战队",
    "startup": "初创企业",
    "dismantled": "捣毁",
    "purge": "清洗",
    "breakthrough": "突破",
    "summit": "峰会",
    "corridor": "走廊",
    "initiative": "倡议",
    "framework": "框架",
    "stakeholder": "利益相关方",
    # Media and journalism terms
    "footage": "视频",
    "decisively": "决定性地",
    "exclusively": "独家",
    "allegedly": "据称",
    "reportedly": "据报道",
    "outlets": "媒体",
    "appropriations": "拨款",
    # Common verbs
    "rebalancing": "再平衡",
    "rallies": "团结",
    "benefiting": "惠及",
    # Additional terms
    "listing": "上市",
    "standoff": "对峙",
    "stalls": "拖延",
    # Scientific/academic terms
    "alpine": "高山",
    "abrupt": "突然的",
    "shift": "变化",
    "abrupt shift": "突变",
    "resilience": "韧性",
}


def _clean_partial_translation(text: str) -> str:
    """Clean up partially translated text.

    Handles patterns like "word（翻译）" by extracting just the Chinese,
    and replaces common untranslated English words with Chinese equivalents.

    Args:
        text: Chinese text that may contain English fragments.

    Returns:
        Cleaned text with English fragments replaced.
    """
    import re

    if not text:
        return text

    # Only applies to text with Chinese characters (avoids modifying pure English)
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    if has_chinese:
        # Pattern 1: English word followed by Chinese translation in parentheses
        # e.g., "capitulation（投降）" -> "投降"
        # e.g., "的 capitulation （投降）" -> "的投降"
        result = re.sub(
            r'\s*([A-Za-z]+(?:-[A-Za-z]+)?)\s*[（(]([^）)]+)[）)]',
            r'\2',
            text
        )
        # Pattern 2: Chinese followed by English abbreviation in parentheses
        # e.g., "电动汽车（EVs）" -> "电动汽车"
        # e.g., "媒体 outlets" -> "媒体" (standalone English after Chinese)
        result = re.sub(
            r'[（(]([A-Za-z]+(?:s)?)[）)]',
            '',
            result
        )
    else:
        result = text

    # Replace common untranslated words (case-insensitive)
    # Use lookahead/lookbehind to match at Chinese-English boundaries too
    for en_word, zh_word in _UNTRANSLATED_WORDS.items():
        # Match English word at word boundaries OR adjacent to Chinese characters
        # (?<![A-Za-z]) = not preceded by letter
        # (?![A-Za-z]) = not followed by letter
        pattern = re.compile(
            r'(?<![A-Za-z])' + re.escape(en_word) + r'(?![A-Za-z])',
            re.IGNORECASE
        )
        result = pattern.sub(zh_word, result)

    return result


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
                # Use original LLM result with cleanup if all else fails
                translated[idx] = _clean_partial_translation(result)
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

    # Final cleanup pass for all Chinese translations
    if check_english:
        cleaned_count = 0
        for i in range(len(translated)):
            if translated[i] != texts[i]:  # Was actually translated
                cleaned = _clean_partial_translation(translated[i])
                if cleaned != translated[i]:
                    translated[i] = cleaned
                    cleaned_count += 1
        if cleaned_count > 0:
            logger.info("Post-processed %d translations to fix English fragments", cleaned_count)

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
