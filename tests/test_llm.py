"""Tests for the LLM client module."""

from __future__ import annotations

from unittest.mock import patch

from analysis.llm import _call_ollama, llm_summarize, llm_translate


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
