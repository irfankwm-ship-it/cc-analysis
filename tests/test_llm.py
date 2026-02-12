"""Tests for the LLM client module."""

from __future__ import annotations

from unittest.mock import patch

from analysis.llm import _call_ollama, _parse_perspectives, llm_summarize, llm_translate


class TestCallOllama:
    """Tests for the low-level ollama API call."""

    def test_successful_call(self) -> None:
        mock_resp = type("Resp", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": "translated text"},
        })()
        with patch("analysis.llm.requests.post", return_value=mock_resp):
            result = _call_ollama("test prompt")
        assert result == "translated text"

    def test_empty_response(self) -> None:
        mock_resp = type("Resp", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": ""},
        })()
        with patch("analysis.llm.requests.post", return_value=mock_resp):
            result = _call_ollama("test prompt")
        assert result is None

    def test_network_error_returns_none(self) -> None:
        with patch("analysis.llm.requests.post", side_effect=ConnectionError("timeout")):
            result = _call_ollama("test prompt")
        assert result is None


class TestLlmTranslate:
    """Tests for LLM translation."""

    def test_translate_zh_to_en(self) -> None:
        with patch("analysis.llm._call_ollama", return_value="China imposed tariffs"):
            result = llm_translate("中国加征关税", "zh", "en")
        assert result == "China imposed tariffs"

    def test_translate_en_to_zh(self) -> None:
        with patch("analysis.llm._call_ollama", return_value="中国加征关税"):
            result = llm_translate("China imposed tariffs", "en", "zh")
        assert result == "中国加征关税"

    def test_translate_returns_none_on_failure(self) -> None:
        with patch("analysis.llm._call_ollama", return_value=None):
            result = llm_translate("some text", "en", "zh")
        assert result is None

    def test_translate_empty_text(self) -> None:
        result = llm_translate("", "en", "zh")
        assert result is None

    def test_translate_rejects_identical_output(self) -> None:
        with patch("analysis.llm._call_ollama", return_value="same text"):
            result = llm_translate("same text", "en", "zh")
        assert result is None


class TestLlmSummarize:
    """Tests for LLM summarization."""

    def test_summarize_returns_shorter_text(self) -> None:
        long_text = "A" * 500
        summary = "Short summary of the article."
        with patch("analysis.llm._call_ollama", return_value=summary):
            result = llm_summarize(long_text, "Test headline")
        assert result == summary

    def test_summarize_rejects_longer_than_input(self) -> None:
        short_text = "Short."
        long_summary = "A" * 100
        with patch("analysis.llm._call_ollama", return_value=long_summary):
            result = llm_summarize(short_text, "Test headline")
        assert result is None

    def test_summarize_returns_none_on_failure(self) -> None:
        with patch("analysis.llm._call_ollama", return_value=None):
            result = llm_summarize("some article text", "headline")
        assert result is None

    def test_summarize_empty_text(self) -> None:
        result = llm_summarize("", "headline")
        assert result is None

    def test_summarize_chinese_prompt(self) -> None:
        long_text = "中" * 500
        summary = "这是中文摘要。"
        with patch("analysis.llm._call_ollama", return_value=summary) as mock:
            result = llm_summarize(long_text, "中文标题", lang="zh")
        assert result == summary
        # Verify Chinese prompt was used
        prompt_text = mock.call_args[0][0]
        assert "请将以下文章总结" in prompt_text


class TestParsePerspectives:
    """Tests for plain-text perspective parsing."""

    def test_english_markers(self) -> None:
        text = (
            "Canadian perspective: Ottawa views this trade deal with concern, "
            "noting potential impacts on agricultural exports. The government "
            "is expected to seek bilateral consultations.\n\n"
            "Beijing perspective: China frames the agreement as mutually "
            "beneficial and consistent with WTO principles. State media "
            "emphasizes open market commitments."
        )
        result = _parse_perspectives(text, "canadian perspective", "beijing perspective", "en")
        assert result is not None
        assert "Ottawa" in result["canada"]
        assert "China" in result["china"]
        assert result["lang"] == "en"

    def test_chinese_markers(self) -> None:
        text = (
            "加拿大视角：渥太华方面对此事高度关注，认为涉及加拿大核心贸易利益。"
            "预计加方将通过外交渠道提出关切。\n\n"
            "北京视角：中方将此定性为正当的经济措施，符合国际贸易规则。"
            "官方媒体强调中国始终坚持互利共赢的合作理念。"
        )
        result = _parse_perspectives(text, "加拿大视角", "北京视角", "zh")
        assert result is not None
        assert "渥太华" in result["canada"]
        assert "中方" in result["china"]
        assert result["lang"] == "zh"

    def test_markers_with_fullwidth_colon(self) -> None:
        text = (
            "Canadian perspective：Ottawa is monitoring the situation closely "
            "and considering diplomatic options for engagement.\n\n"
            "Beijing perspective：Beijing considers this an internal matter "
            "and urges respect for sovereignty."
        )
        result = _parse_perspectives(text, "canadian perspective", "beijing perspective", "en")
        assert result is not None
        assert "Ottawa" in result["canada"]

    def test_missing_marker_returns_none(self) -> None:
        text = "Some text without any perspective markers at all."
        result = _parse_perspectives(text, "canadian perspective", "beijing perspective", "en")
        assert result is None

    def test_too_short_returns_none(self) -> None:
        text = "Canadian perspective: Short.\n\nBeijing perspective: Brief."
        result = _parse_perspectives(text, "canadian perspective", "beijing perspective", "en")
        assert result is None

    def test_reversed_order(self) -> None:
        """Beijing marker appears before Canadian marker."""
        text = (
            "Beijing perspective: China sees this as a routine exercise "
            "of sovereign rights in its territorial waters.\n\n"
            "Canadian perspective: Canada expresses deep concern over "
            "the military activity and its implications for regional stability."
        )
        result = _parse_perspectives(text, "canadian perspective", "beijing perspective", "en")
        assert result is not None
        assert "sovereign" in result["china"]
        assert "concern" in result["canada"]
