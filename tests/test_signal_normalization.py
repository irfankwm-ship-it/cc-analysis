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
    def test_llm_generated(self, mock_llm: object) -> None:
        mock_llm.return_value = {
            "canada": {"en": "LLM Canada", "zh": "LLM加拿大"},
            "china": {"en": "LLM China", "zh": "LLM中国"},
        }
        result = generate_perspectives(
            category="trade",
            is_chinese=False,
            title="Trade news",
            body_text="Long body text about trade",
        )
        assert result.get("llm_generated") is True


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
                "title": {"en": "English title", "zh": "English title"},
                "body": {"en": "English body", "zh": "English body"},
            }
        ]
        result = translate_signals_batch(signals)
        assert len(result) == 1
        mock_translate.assert_called_once()

    def test_preserves_chinese_content(self) -> None:
        signals = [
            {
                "title": {"en": "Translated title", "zh": ""},
                "body": {"en": "Translated body", "zh": ""},
                "_preserved_zh_title": "原始中文标题",
                "_preserved_zh_body": "原始中文内容需要足够长才能通过摘要处理",
            }
        ]
        result = translate_signals_batch(signals)
        assert result[0]["title"]["zh"] == "原始中文标题"
