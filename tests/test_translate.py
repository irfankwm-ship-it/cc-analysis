"""Tests for the translation module."""

from __future__ import annotations

from unittest.mock import patch

from analysis.translate import translate_to_chinese, translate_to_english


class TestTranslateToChinese:
    """Tests for English → Chinese translation."""

    def test_llm_primary_success(self) -> None:
        with patch("analysis.translate.llm_translate", return_value="中文翻译"):
            result = translate_to_chinese(["English text"])
        assert result == ["中文翻译"]

    def test_mymemory_fallback(self) -> None:
        with (
            patch("analysis.translate.llm_translate", return_value=None),
            patch("analysis.translate._mymemory_translate_one", return_value="中文翻译"),
            patch("analysis.translate.time.sleep"),
        ):
            result = translate_to_chinese(["English text"])
        assert result == ["中文翻译"]

    def test_returns_original_on_all_failures(self) -> None:
        with (
            patch("analysis.translate.llm_translate", return_value=None),
            patch("analysis.translate._mymemory_translate_one", return_value=None),
            patch("analysis.translate.time.sleep"),
        ):
            result = translate_to_chinese(["English text"])
        assert result == ["English text"]

    def test_empty_list(self) -> None:
        result = translate_to_chinese([])
        assert result == []

    def test_empty_strings_skipped(self) -> None:
        with patch("analysis.translate.llm_translate", return_value="翻译"):
            result = translate_to_chinese(["text", "", "more"])
        assert result[0] == "翻译"
        assert result[1] == ""
        assert result[2] == "翻译"


class TestTranslateToEnglish:
    """Tests for Chinese → English translation."""

    def test_llm_primary_success(self) -> None:
        with patch("analysis.translate.llm_translate", return_value="English translation"):
            result = translate_to_english(["中文文本"])
        assert result == ["English translation"]

    def test_mymemory_fallback(self) -> None:
        with (
            patch("analysis.translate.llm_translate", return_value=None),
            patch("analysis.translate._mymemory_translate_one", return_value="English text"),
            patch("analysis.translate.time.sleep"),
        ):
            result = translate_to_english(["中文文本"])
        assert result == ["English text"]

    def test_returns_original_on_failure(self) -> None:
        with (
            patch("analysis.translate.llm_translate", return_value=None),
            patch("analysis.translate._mymemory_translate_one", return_value=None),
            patch("analysis.translate.time.sleep"),
        ):
            result = translate_to_english(["中文文本"])
        assert result == ["中文文本"]
