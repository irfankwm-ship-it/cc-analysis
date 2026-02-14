"""Tests for the LLM client module."""

from __future__ import annotations

from unittest.mock import patch

from analysis.llm import (
    _call_ollama,
    _parse_perspectives,
    _strip_prompt_artifacts,
    llm_summarize,
    llm_translate,
)


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

    def test_ottawa_beijing_markers(self) -> None:
        """New OTTAWA/BEIJING markers parse correctly."""
        text = (
            "OTTAWA perspective: This trade restriction directly impacts "
            "Canadian canola exporters and Ottawa may seek WTO arbitration.\n\n"
            "BEIJING perspective: China frames the restriction as a food "
            "safety measure consistent with its regulatory sovereignty."
        )
        result = _parse_perspectives(text, "ottawa perspective", "beijing perspective", "en")
        assert result is not None
        assert "canola" in result["canada"]
        assert "sovereignty" in result["china"]

    def test_zh_ottawa_beijing_markers(self) -> None:
        """New 渥太华视角/北京视角 markers parse correctly."""
        text = (
            "渥太华视角：加拿大油菜籽出口商直接受到影响，渥太华可能寻求世贸组织仲裁。\n\n"
            "北京视角：中方将此限制定性为符合监管主权的食品安全措施。"
        )
        result = _parse_perspectives(text, "渥太华视角", "北京视角", "zh")
        assert result is not None
        assert "渥太华" in result["canada"]
        assert "中方" in result["china"]

    def test_short_ottawa_beijing_markers(self) -> None:
        """Short OTTAWA/BEIJING markers (without 'perspective') parse correctly."""
        text = (
            "OTTAWA: This trade restriction directly impacts "
            "Canadian canola exporters and Ottawa may seek WTO arbitration.\n\n"
            "BEIJING: China frames the restriction as a food "
            "safety measure consistent with its regulatory sovereignty."
        )
        result = _parse_perspectives(text, "ottawa", "beijing", "en")
        assert result is not None
        assert "canola" in result["canada"]
        assert "sovereignty" in result["china"]

    def test_short_zh_markers(self) -> None:
        """Short 渥太华/北京 markers (without 视角) parse correctly."""
        text = (
            "渥太华：加拿大油菜籽出口商直接受到影响，渥太华可能寻求世贸组织仲裁。\n\n"
            "北京：中方将此限制定性为符合监管主权的食品安全措施。"
        )
        result = _parse_perspectives(text, "渥太华", "北京", "zh")
        assert result is not None
        assert "渥太华" in result["canada"]
        assert "中方" in result["china"]


class TestStripPromptArtifacts:
    """Tests for _strip_prompt_artifacts."""

    def test_strips_en_parenthetical_prefix(self) -> None:
        text = "(Pragmatic, Canadian interests, values-aware): Ottawa is concerned about the tariff impact."
        result = _strip_prompt_artifacts(text)
        assert result == "Ottawa is concerned about the tariff impact."

    def test_strips_en_uppercase_parenthetical(self) -> None:
        text = "(PRAGMATIC, CANADIAN INTERESTS): Ottawa should consider trade options."
        result = _strip_prompt_artifacts(text)
        assert result == "Ottawa should consider trade options."

    def test_strips_sovereignty_prefix(self) -> None:
        text = "(Sovereignty-focused, official framing): China considers this matter internal."
        result = _strip_prompt_artifacts(text)
        assert result == "China considers this matter internal."

    def test_strips_zh_parenthetical(self) -> None:
        text = "（务实、加拿大利益优先）：加拿大对此表示关切。"
        result = _strip_prompt_artifacts(text)
        assert result == "加拿大对此表示关切。"

    def test_strips_zh_sovereignty_parenthetical(self) -> None:
        text = "（主权优先、官方定调）：中方认为这是内政问题。"
        result = _strip_prompt_artifacts(text)
        assert result == "中方认为这是内政问题。"

    def test_strips_question_form_en(self) -> None:
        text = "How does this affect Canada specifically? Ottawa views the tariff increase with alarm."
        result = _strip_prompt_artifacts(text)
        assert result == "Ottawa views the tariff increase with alarm."

    def test_strips_question_form_zh(self) -> None:
        text = "这对加拿大有什么具体影响？加拿大方面高度关注关税变化。"
        result = _strip_prompt_artifacts(text)
        assert result == "加拿大方面高度关注关税变化。"

    def test_strips_rules_block(self) -> None:
        text = "Ottawa is concerned about the situation.\n\nRULES:\n- Must reference facts\n- Must differ"
        result = _strip_prompt_artifacts(text)
        assert result == "Ottawa is concerned about the situation."

    def test_strips_zh_rules_block(self) -> None:
        text = "中方对此表示反对。\n\n规则：\n- 必须引用具体事实\n- 两个视角必须不同"
        result = _strip_prompt_artifacts(text)
        assert result == "中方对此表示反对。"

    def test_strips_perspective_label_en(self) -> None:
        text = "Perspective: Canada will benefit from this transaction by promoting cooperation."
        result = _strip_prompt_artifacts(text)
        assert result == "Canada will benefit from this transaction by promoting cooperation."

    def test_strips_perspective_label_en_newline(self) -> None:
        text = "Perspective:\nCanada has traditionally been pragmatic in its approach."
        result = _strip_prompt_artifacts(text)
        assert result == "Canada has traditionally been pragmatic in its approach."

    def test_strips_perspective_label_zh(self) -> None:
        text = "视角：加拿大将从这种交易中受益。"
        result = _strip_prompt_artifacts(text)
        assert result == "加拿大将从这种交易中受益。"

    def test_strips_perspective_label_zh_newline(self) -> None:
        text = "视角：\n加拿大在国际事务中一向秉持务实的态度。"
        result = _strip_prompt_artifacts(text)
        assert result == "加拿大在国际事务中一向秉持务实的态度。"

    def test_strips_instruction_text_en(self) -> None:
        text = "Pragmatic, Canadian interests first, values focus:\n- The operation of the company is key."
        result = _strip_prompt_artifacts(text)
        assert result == "- The operation of the company is key."

    def test_strips_instruction_text_zh(self) -> None:
        text = "务实、加拿大利益优先、关注价值观：\n这起事件反映了教育体系中的问题。"
        result = _strip_prompt_artifacts(text)
        assert result == "这起事件反映了教育体系中的问题。"

    def test_strips_instruction_description_en(self) -> None:
        text = "Prudent and pragmatic in political stance, focused on actual operations in national interests."
        result = _strip_prompt_artifacts(text)
        assert result == ""

    def test_strips_instruction_description_zh(self) -> None:
        text = "- 政治立场稳健务实，专注于国家利益和国际事务中的实际操作。"
        result = _strip_prompt_artifacts(text)
        assert result == ""

    def test_strips_combined_perspective_plus_instruction(self) -> None:
        text = "Perspective: Pragmatic, Canadian interests first, values focus:\n- The semiconductor supply chain is critical."
        result = _strip_prompt_artifacts(text)
        assert result == "- The semiconductor supply chain is critical."

    def test_leaves_clean_text_unchanged(self) -> None:
        text = "Ottawa is closely monitoring the trade dispute and considering diplomatic options."
        result = _strip_prompt_artifacts(text)
        assert result == text

    def test_leaves_clean_zh_unchanged(self) -> None:
        text = "加拿大密切关注贸易争端，正在考虑外交选项。"
        result = _strip_prompt_artifacts(text)
        assert result == text

    def test_integration_with_parse_perspectives(self) -> None:
        """Artifact stripping works inside _parse_perspectives."""
        text = (
            "OTTAWA: (Pragmatic, Canadian interests): "
            "Ottawa views the tariff escalation with deep concern, "
            "as Canadian canola exports face significant disruption.\n\n"
            "BEIJING: (Sovereignty-focused, official framing): "
            "China frames these measures as legitimate food safety enforcement "
            "consistent with its sovereign regulatory authority."
        )
        result = _parse_perspectives(text, "ottawa", "beijing", "en")
        assert result is not None
        # Should not contain the parenthetical prefix
        assert "(Pragmatic" not in result["canada"]
        assert "(Sovereignty" not in result["china"]
        # Should still contain the actual perspective content
        assert "canola" in result["canada"]
        assert "sovereign" in result["china"]
