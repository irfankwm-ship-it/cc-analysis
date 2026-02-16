"""Tests for the translation module."""

from __future__ import annotations

from unittest.mock import patch

from analysis.translate import (
    _clean_partial_translation,
    _contains_untranslated_english,
    _strip_translation_preamble,
    fix_english_text,
    fix_gender_pronouns,
    fix_name_hallucinations,
    translate_to_chinese,
    translate_to_english,
)


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

    def test_applies_pronoun_correction(self) -> None:
        with patch(
            "analysis.translate.llm_translate",
            return_value="Pelosi said he will visit Taiwan",
        ):
            result = translate_to_english(["佩洛西说他将访问台湾"])
        assert "she" in result[0].lower()


class TestFixGenderPronouns:
    """Tests for pronoun correction on known figures."""

    def test_corrects_pelosi(self) -> None:
        text = "Pelosi said he would visit Taiwan."
        result = fix_gender_pronouns(text)
        assert "she" in result.lower()
        assert "he " not in result.lower().replace("she", "")

    def test_corrects_his_to_her(self) -> None:
        text = "Tsai Ing-wen announced his new policy."
        result = fix_gender_pronouns(text)
        assert "her" in result.lower()

    def test_no_change_for_male_figures(self) -> None:
        text = "Xi Jinping said he would increase tariffs."
        result = fix_gender_pronouns(text)
        assert result == text

    def test_empty_text(self) -> None:
        assert fix_gender_pronouns("") == ""

    def test_no_known_figures(self) -> None:
        text = "The minister said he would resign."
        result = fix_gender_pronouns(text)
        assert result == text

    def test_preserves_surrounding_text(self) -> None:
        text = "In Ottawa, Freeland presented his budget plan to parliament."
        result = fix_gender_pronouns(text)
        assert "her" in result.lower()
        assert "Ottawa" in result
        assert "parliament" in result


class TestContainsUntranslatedEnglish:
    """Tests for detecting untranslated English in Chinese text."""

    def test_clean_chinese(self) -> None:
        assert not _contains_untranslated_english("中国宣布新贸易政策")

    def test_english_fragments(self) -> None:
        assert _contains_untranslated_english(
            "中国 announced new trade policy 关于贸易"
        )

    def test_empty_string(self) -> None:
        assert not _contains_untranslated_english("")

    def test_mostly_english(self) -> None:
        assert _contains_untranslated_english("This is mostly English text 中文")

    def test_custom_threshold(self) -> None:
        # Low threshold should flag even small amounts of English
        text = "中国贸易政策ABC的发展"
        assert not _contains_untranslated_english(text, threshold=0.5)

    def test_punctuation_only(self) -> None:
        assert not _contains_untranslated_english("...!!!")


class TestCleanPartialTranslation:
    """Tests for cleaning up partially translated text."""

    def test_removes_english_with_chinese_parens(self) -> None:
        text = "capitulation（投降）已经开始"
        result = _clean_partial_translation(text)
        assert "capitulation" not in result
        assert "投降" in result

    def test_removes_english_abbreviations(self) -> None:
        text = "电动汽车（EVs）行业"
        result = _clean_partial_translation(text)
        assert "EVs" not in result
        assert "电动汽车" in result

    def test_replaces_untranslated_words(self) -> None:
        text = "这是一个unprecedented的举措"
        result = _clean_partial_translation(text)
        assert "unprecedented" not in result
        assert "史无前例" in result

    def test_empty_text(self) -> None:
        assert _clean_partial_translation("") == ""

    def test_pure_english_unchanged(self) -> None:
        text = "This is pure English text with sanctions"
        result = _clean_partial_translation(text)
        # Should still replace known words even in English
        assert "制裁" in result

    def test_multiple_replacements(self) -> None:
        text = "中国的sanctions和tariffs政策"
        result = _clean_partial_translation(text)
        assert "制裁" in result
        assert "关税" in result


class TestStripTranslationPreamble:
    """Tests for stripping LLM preambles from translations."""

    def test_heres_the_translation_from(self) -> None:
        text = 'Here\'s the translation from Simplified Chinese to English: "AI Weekly Report"'
        result = _strip_translation_preamble(text)
        assert result == "AI Weekly Report"

    def test_heres_the_translation_colon(self) -> None:
        text = "Here's the translation: Canada imposes sanctions"
        result = _strip_translation_preamble(text)
        assert result == "Canada imposes sanctions"

    def test_here_is_the_translation(self) -> None:
        text = "Here is the translation from Chinese to English: Trade war escalates"
        result = _strip_translation_preamble(text)
        assert result == "Trade war escalates"

    def test_translation_colon(self) -> None:
        text = "Translation: Beijing responds to tariffs"
        result = _strip_translation_preamble(text)
        assert result == "Beijing responds to tariffs"

    def test_translated_text_colon(self) -> None:
        text = "Translated text: Xi Jinping meets delegation"
        result = _strip_translation_preamble(text)
        assert result == "Xi Jinping meets delegation"

    def test_in_english_colon(self) -> None:
        text = "In English: China grants zero-tariff treatment"
        result = _strip_translation_preamble(text)
        assert result == "China grants zero-tariff treatment"

    def test_no_preamble_unchanged(self) -> None:
        text = "Canada imposes new sanctions on China"
        result = _strip_translation_preamble(text)
        assert result == text

    def test_empty_string(self) -> None:
        assert _strip_translation_preamble("") == ""

    def test_none_returns_none(self) -> None:
        assert _strip_translation_preamble(None) is None

    def test_strips_surrounding_quotes(self) -> None:
        text = '"AI Weekly: DouBan Big Model 2.0 released"'
        result = _strip_translation_preamble(text)
        assert result == "AI Weekly: DouBan Big Model 2.0 released"

    def test_curly_quotes_stripped(self) -> None:
        text = '\u201cTrade agreement signed\u201d'
        result = _strip_translation_preamble(text)
        assert result == "Trade agreement signed"

    def test_preamble_plus_quotes(self) -> None:
        text = 'Here\'s the translation from Simplified Chinese to English: "AI Weekly: DouBan Big Model 2.0 released; Zhipu GLM-5 deeply optimized"'
        result = _strip_translation_preamble(text)
        assert result == "AI Weekly: DouBan Big Model 2.0 released; Zhipu GLM-5 deeply optimized"


class TestFixNameHallucinations:
    """Tests for name hallucination correction (Fix 4)."""

    def test_trudeau_to_carney_english(self) -> None:
        text = "Trudeau announced new trade sanctions against China."
        result = fix_name_hallucinations(text, "en")
        assert "Carney" in result
        assert "Trudeau" not in result

    def test_justin_trudeau_to_mark_carney_english(self) -> None:
        text = "Justin Trudeau met with Xi Jinping at the summit."
        result = fix_name_hallucinations(text, "en")
        assert "Mark Carney" in result
        assert "Justin Trudeau" not in result

    def test_trudeau_to_carney_chinese(self) -> None:
        text = "特鲁多宣布了新的贸易制裁。"
        result = fix_name_hallucinations(text, "zh")
        assert "卡尼" in result
        assert "特鲁多" not in result

    def test_full_chinese_name_correction(self) -> None:
        text = "贾斯廷·特鲁多与习近平在峰会上会面。"
        result = fix_name_hallucinations(text, "zh")
        assert "马克·卡尼" in result
        assert "贾斯廷·特鲁多" not in result

    def test_case_insensitive_english(self) -> None:
        text = "TRUDEAU said he would impose tariffs."
        result = fix_name_hallucinations(text, "en")
        assert "Carney" in result

    def test_no_change_for_unrelated_text(self) -> None:
        text = "Xi Jinping addressed the conference."
        result = fix_name_hallucinations(text, "en")
        assert result == text

    def test_empty_text(self) -> None:
        assert fix_name_hallucinations("", "en") == ""

    def test_applied_in_translate_to_english(self) -> None:
        with patch(
            "analysis.translate.llm_translate",
            return_value="Trudeau imposed sanctions",
        ):
            result = translate_to_english(["特鲁多实施制裁"])
        assert "Carney" in result[0]
        assert "Trudeau" not in result[0]

    def test_applied_in_translate_to_chinese(self) -> None:
        with patch(
            "analysis.translate.llm_translate",
            return_value="特鲁多宣布新政策",
        ):
            result = translate_to_chinese(["Trudeau announces new policy"])
        assert "卡尼" in result[0]
        assert "特鲁多" not in result[0]

    def test_applied_in_fix_english_text(self) -> None:
        text = "Trudeau met with Chinese delegation."
        result = fix_english_text(text)
        assert "Carney" in result
        assert "Trudeau" not in result


class TestStrictRetryPath:
    """Tests for the strict translation retry path."""

    def test_strict_retry_on_english_fragments(self) -> None:
        with (
            patch(
                "analysis.translate.llm_translate",
                return_value="中国的escalation政策",
            ),
            patch(
                "analysis.translate.llm_translate_strict",
                return_value="中国的升级政策",
            ),
        ):
            result = translate_to_chinese(["China's escalation policy"])
        assert "升级" in result[0]

    def test_mymemory_fallback_after_strict_fails(self) -> None:
        with (
            patch(
                "analysis.translate.llm_translate",
                return_value="中国的escalation政策",
            ),
            patch(
                "analysis.translate.llm_translate_strict",
                return_value="中国的still escalation",
            ),
            patch(
                "analysis.translate._mymemory_translate_one",
                return_value="中国的升级政策",
            ),
        ):
            result = translate_to_chinese(["China's escalation policy"])
        assert "升级" in result[0]

    def test_cleanup_fallback_when_all_retries_fail(self) -> None:
        with (
            patch(
                "analysis.translate.llm_translate",
                return_value="中国的sanctions政策",
            ),
            patch(
                "analysis.translate.llm_translate_strict",
                return_value="中国的still sanctions here",
            ),
            patch(
                "analysis.translate._mymemory_translate_one",
                return_value="中国的more sanctions text",
            ),
        ):
            result = translate_to_chinese(["China's sanctions policy"])
        # Should clean up the original LLM result via _clean_partial_translation
        assert "制裁" in result[0]
