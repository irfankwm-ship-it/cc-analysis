"""Tests for signal_normalization module."""

from __future__ import annotations

from unittest.mock import patch

from analysis.signal_normalization import (
    extract_quote,
    generate_implications,
    generate_perspectives,
    has_english_fragments,
    is_primarily_chinese,
    to_bilingual,
    translate_signals_batch,
)


class TestToBilingual:
    def test_string_value(self) -> None:
        result = to_bilingual("Hello")
        assert result == {"en": "Hello", "zh": "Hello"}

    def test_dict_value(self) -> None:
        result = to_bilingual({"en": "Hello", "zh": "你好"})
        assert result == {"en": "Hello", "zh": "你好"}

    def test_none_value(self) -> None:
        result = to_bilingual(None)
        assert result == {"en": "", "zh": ""}


class TestGenerateImplications:
    def test_diplomatic_critical(self) -> None:
        result = generate_implications(
            "diplomatic", "critical",
            impact_templates={
                "diplomatic": {"en": "Impact text", "zh": "影响"},
            },
            watch_templates={
                "critical": {
                    "en": {"diplomatic": "Watch critical"},
                    "zh": {"diplomatic": "关注危机"},
                },
                "default": {
                    "en": {"diplomatic": "Monitor"},
                    "zh": {"diplomatic": "跟踪"},
                },
            },
        )
        assert "canada_impact" in result
        assert "what_to_watch" in result
        assert result["what_to_watch"]["en"] == "Watch critical"

    def test_default_severity(self) -> None:
        result = generate_implications(
            "trade", "moderate",
            impact_templates={
                "trade": {"en": "Trade impact", "zh": "贸易影响"},
            },
            watch_templates={
                "default": {
                    "en": {"trade": "Monitor trade"},
                    "zh": {"trade": "跟踪贸易"},
                },
            },
        )
        assert result["what_to_watch"]["en"] == "Monitor trade"


class TestExtractQuote:
    def test_finds_quote(self) -> None:
        text = "The minister said the policy will change. Other details followed."
        result = extract_quote(text, ["said", "stated"])
        assert result is not None
        assert "said" in result.lower()

    def test_no_match(self) -> None:
        text = "Short text without indicators."
        result = extract_quote(text, ["said", "stated"])
        assert result is None

    def test_empty_text(self) -> None:
        assert extract_quote("", ["said"]) is None


class TestGeneratePerspectives:
    @patch("analysis.signal_normalization.llm_generate_perspectives", return_value=None)
    def test_template_fallback(self, mock_llm: object) -> None:
        result = generate_perspectives(
            category="trade",
            is_chinese=False,
            canada_perspective={"trade": {"en": "Canada view", "zh": "加拿大观点"}},
            china_perspective={"trade": {"en": "China view", "zh": "中国观点"}},
        )
        assert "primary_source" in result
        assert result["canada"]["en"] == "Canada view"
        assert result["china"]["en"] == "China view"

    @patch("analysis.signal_normalization.llm_generate_perspectives")
    def test_llm_generated_english(self, mock_llm: object) -> None:
        """LLM returns English-only perspectives for English source."""
        mock_llm.return_value = {
            "canada": "Ottawa views this trade move with concern.",
            "china": "Beijing frames this as a win-win outcome.",
            "lang": "en",
        }
        result = generate_perspectives(
            category="trade",
            is_chinese=False,
            title="Trade news",
            body_text="Long body text about trade developments and policy",
            lang="en",
        )
        assert result.get("llm_generated") is True
        assert result["canada"]["en"] == "Ottawa views this trade move with concern."
        assert result["canada"]["zh"] == ""  # empty — filled later by translation
        assert result["china"]["en"] == "Beijing frames this as a win-win outcome."

    @patch("analysis.signal_normalization.llm_generate_perspectives")
    def test_llm_generated_chinese(self, mock_llm: object) -> None:
        """LLM returns Chinese-only perspectives for Chinese source."""
        mock_llm.return_value = {
            "canada": "渥太华对此表示关切，认为涉及加拿大利益。",
            "china": "北京方面将此定性为正常的主权行为。",
            "lang": "zh",
        }
        result = generate_perspectives(
            category="diplomatic",
            is_chinese=True,
            title="外交部发言人就加中关系问题答记者问",
            body_text="外交部发言人表示，中方一贯主张通过对话协商解决分歧",
            lang="zh",
        )
        assert result.get("llm_generated") is True
        assert result["canada"]["zh"] == "渥太华对此表示关切，认为涉及加拿大利益。"
        assert result["canada"]["en"] == ""  # empty — filled later by translation
        assert result["china"]["zh"] == "北京方面将此定性为正常的主权行为。"

    @patch("analysis.signal_normalization.llm_generate_perspectives", return_value=None)
    def test_llm_failure_falls_back_to_template(self, mock_llm: object) -> None:
        """When LLM returns None, perspectives fall back to category templates."""
        result = generate_perspectives(
            category="military",
            is_chinese=False,
            title="PLA exercise",
            body_text="Military exercise in the region",
            lang="en",
            canada_perspective={"military": {"en": "Canada mil view", "zh": "加军事观点"}},
            china_perspective={"military": {"en": "China mil view", "zh": "中军事观点"}},
        )
        assert result.get("llm_generated") is not True
        assert result["canada"]["en"] == "Canada mil view"


class TestHasEnglishFragments:
    def test_pure_chinese(self) -> None:
        assert not has_english_fragments("这是中文文本没有英文")

    def test_mixed_text(self) -> None:
        assert has_english_fragments("中文 with lots of English words mixed in here")

    def test_empty_text(self) -> None:
        assert not has_english_fragments("")


class TestIsPrimarilyChinese:
    def test_chinese_text(self) -> None:
        assert is_primarily_chinese("这是中文文本")

    def test_english_text(self) -> None:
        assert not is_primarily_chinese("This is English text")

    def test_empty(self) -> None:
        assert not is_primarily_chinese("")


class TestTranslateSignalsBatch:
    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_translates_english_signals(self, mock_translate: object) -> None:
        mock_translate.return_value = ["中文标题", "中文内容"]
        signals = [
            {
                "title": {"en": "English title", "zh": ""},
                "body": {"en": "English body", "zh": ""},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        assert len(result) == 1
        mock_translate.assert_called_once()
        assert result[0]["title"]["zh"] == "中文标题"
        assert result[0]["body"]["zh"] == "中文内容"

    @patch("analysis.signal_normalization.translate_to_english")
    def test_translates_chinese_signals(self, mock_translate: object) -> None:
        mock_translate.return_value = ["English title", "English body"]
        signals = [
            {
                "title": {"en": "", "zh": "中文标题"},
                "body": {"en": "", "zh": "中文内容"},
                "_source_lang": "zh",
            }
        ]
        result = translate_signals_batch(signals)
        assert len(result) == 1
        mock_translate.assert_called_once()
        assert result[0]["title"]["en"] == "English title"
        assert result[0]["body"]["en"] == "English body"

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_translates_perspectives(self, mock_translate: object) -> None:
        """Perspectives generated in English get translated to Chinese."""
        mock_translate.return_value = [
            "中文标题", "中文内容",
            "渥太华对此关切", "北京认为正当",
        ]
        signals = [
            {
                "title": {"en": "Trade news", "zh": ""},
                "body": {"en": "Trade body", "zh": ""},
                "perspectives": {
                    "canada": {"en": "Ottawa concerned", "zh": ""},
                    "china": {"en": "Beijing justified", "zh": ""},
                    "llm_generated": True,
                },
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        assert len(result) == 1
        assert result[0]["perspectives"]["canada"]["zh"] == "渥太华对此关切"
        assert result[0]["perspectives"]["china"]["zh"] == "北京认为正当"

    @patch("analysis.signal_normalization.translate_to_english")
    def test_translates_chinese_perspectives(self, mock_translate: object) -> None:
        """Perspectives generated in Chinese get translated to English."""
        mock_translate.return_value = [
            "English title", "English body",
            "Ottawa is concerned", "Beijing sees this as justified",
        ]
        signals = [
            {
                "title": {"en": "", "zh": "贸易新闻"},
                "body": {"en": "", "zh": "贸易内容"},
                "perspectives": {
                    "canada": {"en": "", "zh": "渥太华对此关切"},
                    "china": {"en": "", "zh": "北京认为此举正当"},
                    "llm_generated": True,
                },
                "_source_lang": "zh",
            }
        ]
        result = translate_signals_batch(signals)
        assert len(result) == 1
        assert result[0]["perspectives"]["canada"]["en"] == "Ottawa is concerned"
        assert result[0]["perspectives"]["china"]["en"] == "Beijing sees this as justified"

    def test_skips_already_bilingual(self) -> None:
        """Signals with both languages filled are left alone."""
        signals = [
            {
                "title": {"en": "Title", "zh": "标题"},
                "body": {"en": "Body", "zh": "内容"},
                "perspectives": {
                    "canada": {"en": "Canada", "zh": "加拿大"},
                    "china": {"en": "China", "zh": "中国"},
                },
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        assert result[0]["title"]["en"] == "Title"
        assert result[0]["title"]["zh"] == "标题"
