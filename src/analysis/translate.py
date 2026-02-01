"""Chinese translation via MyMemory Translation API.

Provides translation of English texts to Simplified Chinese.
Gracefully falls back to returning original text if the API call fails.

Uses the free MyMemory API (api.mymemory.translated.net).
Quota: ~5000 chars/day anonymous, higher with email/key.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.mymemory.translated.net/get"


def _translate_one(text: str) -> str | None:
    """Translate a single English text to Simplified Chinese.

    Returns translated text, or None on failure.
    """
    try:
        resp = requests.get(
            _BASE_URL,
            params={
                "langpair": "en|zh-CN",
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


def translate_to_chinese(texts: list[str]) -> list[str]:
    """Translate a list of English texts to Simplified Chinese.

    Makes one API call per text with a small delay to respect rate limits.
    Returns original texts for any that fail.
    """
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not non_empty:
        return texts

    translated = list(texts)  # copy
    success = 0
    quota_done = False

    for idx, text in non_empty:
        if quota_done:
            break
        result = _translate_one(text)
        if result:
            translated[idx] = result
            success += 1
        elif result is None:
            # Check if it was a quota issue (logged inside _translate_one)
            pass
        # Small delay between requests to respect rate limits
        time.sleep(0.2)

    logger.info("MyMemory: translated %d/%d texts to Chinese", success, len(non_empty))
    return translated
