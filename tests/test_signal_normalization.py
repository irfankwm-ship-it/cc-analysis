"""Tests for signal_normalization module."""

from __future__ import annotations

from unittest.mock import patch

from analysis.signal_normalization import (
    _validate_summary,
    extract_quote,
    generate_implications,
    generate_perspectives,
    has_canada_nexus,
    has_english_fragments,
    is_primarily_chinese,
    score_perspective_relevance,
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
            title="Canada-China trade dispute",
            body_text="Ottawa and Beijing clash over tariff policy",
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
            title="Canada-China trade news",
            body_text="Ottawa and Beijing discuss trade developments and tariff policy",
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
            body_text="外交部发言人表示，中方一贯主张通过对话协商解决加拿大与中国的分歧",
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
            title="China PLA exercise near Taiwan",
            body_text="Military exercise in the Indo-Pacific region by Chinese forces",
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

    def test_strips_cjk_from_en_title(self) -> None:
        """CJK characters leftover in EN text should be stripped."""
        signals = [
            {
                "title": {"en": "exports $3.5B 创历史新高", "zh": "出口35亿美元创历史新高"},
                "body": {"en": "Body text", "zh": "内容"},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        assert result[0]["title"]["en"] == "exports $3.5B"

    def test_strips_cjk_mid_sentence(self) -> None:
        """CJK fragments in the middle of EN text should be stripped cleanly."""
        signals = [
            {
                "title": {"en": "Gao wins, 预计 she will anger", "zh": "高胜利，预计她会愤怒"},
                "body": {"en": "Body", "zh": "内容"},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        assert result[0]["title"]["en"] == "Gao wins, she will anger"

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_zh_empty_title_gets_en_fallback(self, mock_translate: object) -> None:
        """Empty ZH title should get EN text as fallback (title only)."""
        # Translation returns empty strings (simulating failure)
        mock_translate.return_value = ["", ""]
        signals = [
            {
                "title": {"en": "English title", "zh": ""},
                "body": {"en": "English body", "zh": ""},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        # Title gets EN fallback (blank title is worse than wrong language)
        assert result[0]["title"]["zh"] == "English title"
        # Body does NOT get EN fallback (blank is better than wrong language)
        assert result[0]["body"]["zh"] == ""

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_zh_all_english_gets_fallback(self, mock_translate: object) -> None:
        """ZH field with only English text should get EN fallback."""
        # Translation returns English (bad translation)
        mock_translate.return_value = ["Just English words here"]
        signals = [
            {
                "title": {"en": "Real English title", "zh": "Just English words here"},
                "body": {"en": "Body", "zh": "正文内容"},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        # Title ZH has no CJK characters, so should be replaced by EN
        assert result[0]["title"]["zh"] == "Real English title"
        # Body ZH has CJK characters, so should be untouched
        assert result[0]["body"]["zh"] == "正文内容"


class TestScorePerspectiveRelevance:
    def test_ottawa_score_high_for_canada_signal(self) -> None:
        score = score_perspective_relevance(
            "Canada sanctions Chinese officials", "Ottawa imposed new sanctions"
        )
        assert score["ottawa"] >= 3

    def test_ottawa_score_low_for_unrelated_signal(self) -> None:
        score = score_perspective_relevance(
            "African zero-tariff treatment",
            "China grants zero-tariff to African countries under new policy",
        )
        assert score["ottawa"] < 3

    def test_beijing_score_high_for_china_signal(self) -> None:
        score = score_perspective_relevance(
            "Xi Jinping mentions Taiwan", "Beijing reaffirms one-China principle"
        )
        assert score["beijing"] >= 3

    def test_beijing_score_low_for_no_china_content(self) -> None:
        """Signal with no China references should score low for Beijing."""
        score = score_perspective_relevance(
            "Local weather update", "Rain expected in the area this weekend"
        )
        assert score["beijing"] < 3

    def test_tier2_alliance_keywords_score(self) -> None:
        """Alliance keywords should give moderate Ottawa score."""
        score = score_perspective_relevance(
            "NATO discusses Indo-Pacific", "Five Eyes intelligence alliance meets"
        )
        assert score["ottawa"] >= 3

    def test_combined_tiers_accumulate(self) -> None:
        """A signal matching multiple tiers should accumulate scores."""
        score = score_perspective_relevance(
            "Canada imposes tariff on Chinese imports",
            "Ottawa targets supply chain for critical minerals from Beijing",
        )
        # tier1 (canada=5) + tier3 (tariff=1) = 6+ for ottawa
        assert score["ottawa"] >= 6
        # tier1 (chinese=5) for beijing
        assert score["beijing"] >= 5

    def test_chinese_keywords_work(self) -> None:
        """Chinese-language keywords should match correctly."""
        score = score_perspective_relevance("加拿大制裁", "渥太华宣布对中国实施新制裁")
        assert score["ottawa"] >= 5
        assert score["beijing"] >= 5


class TestPerspectiveScoringIntegration:
    @patch("analysis.signal_normalization.llm_generate_perspectives", return_value=None)
    def test_no_impact_fallback_used_for_low_ottawa(self, mock_llm: object) -> None:
        """Signal with no Canada nexus should get no_impact template for Ottawa."""
        result = generate_perspectives(
            category="social",
            is_chinese=False,
            body_text="China grants zero-tariff status to African countries under new trade policy",
            title="Zero-tariff for Africa",
            lang="en",
        )
        # Ottawa score should be below threshold, so no_impact template used
        assert result.get("ottawa_score", 99) < 3
        assert "no significant" in result["canada"]["en"].lower()

    @patch("analysis.signal_normalization.llm_generate_perspectives", return_value=None)
    def test_normal_template_used_for_high_ottawa(self, mock_llm: object) -> None:
        """Signal with Canada nexus should use category template, not no_impact."""
        result = generate_perspectives(
            category="trade",
            is_chinese=False,
            body_text="Canada and China reach trade agreement on canola exports",
            title="Canada-China canola deal",
            lang="en",
            canada_perspective={"trade": {"en": "Canada trade view", "zh": "加贸易"}},
            china_perspective={"trade": {"en": "China trade view", "zh": "中贸易"}},
        )
        assert result.get("ottawa_score", 0) >= 3
        # Should NOT be the no_impact template
        assert "no significant" not in result["canada"]["en"].lower()

    @patch("analysis.signal_normalization.llm_generate_perspectives")
    def test_beijing_only_mode(self, mock_llm: object) -> None:
        """When Ottawa is below threshold, LLM should be called in beijing_only mode."""
        mock_llm.return_value = {
            "canada": "",
            "china": "Beijing views this African trade deal as advancing South-South cooperation.",
            "lang": "en",
        }
        result = generate_perspectives(
            category="trade",
            is_chinese=False,
            body_text="China grants zero-tariff status to African nations as Belt and Road expands",
            title="China-Africa zero tariff deal",
            lang="en",
        )
        # LLM called with beijing_only mode
        call_args = mock_llm.call_args
        assert call_args.kwargs.get("perspective_mode") == "beijing_only"
        # Ottawa gets no_impact, Beijing gets LLM result
        assert "no significant" in result["canada"]["en"].lower()
        assert result.get("llm_generated") is True


class TestT2SNormalizationInScoring:
    """Tests for Fix 1: Traditional Chinese keyword matching via T2S conversion."""

    def test_traditional_chinese_taiwan_keywords_match(self) -> None:
        """Traditional Chinese 臺灣 should match Simplified 台湾 in beijing keywords."""
        # 臺灣 is Traditional for Taiwan; 台湾 is Simplified (in _BEIJING_KEYWORDS)
        score = score_perspective_relevance("臺灣問題", "中國大陸與臺灣的關係日趨緊張")
        assert score["beijing"] >= 4  # tier2 match on 台湾

    def test_traditional_chinese_canada_keywords_match(self) -> None:
        """Traditional Chinese 加拿大 should match in Ottawa keywords."""
        score = score_perspective_relevance("加拿大貿易政策", "渥太華討論新的外交方案")
        assert score["ottawa"] >= 5  # tier1 match on 加拿大/渥太华

    def test_has_canada_nexus_with_traditional_chinese(self) -> None:
        """has_canada_nexus should match Traditional Chinese references."""
        assert has_canada_nexus("加拿大與臺灣", "渥太華宣佈新政策")

    def test_simplified_still_works(self) -> None:
        """Existing Simplified Chinese scoring should be unaffected."""
        score = score_perspective_relevance("台湾问题", "中国大陆与台湾关系紧张")
        assert score["beijing"] >= 4


class TestValidateSummary:
    """Tests for Fix 2: LLM summary topical relevance validation."""

    def test_relevant_english_summary_passes(self) -> None:
        title = "Canada imposes tariffs on Chinese steel"
        body = "Ottawa announced new tariffs on steel imports from China, affecting billions in trade."
        summary = "Canada announced tariffs on Chinese steel imports affecting trade."
        assert _validate_summary(summary, title, body, "en") is True

    def test_fabricated_english_summary_rejected(self) -> None:
        title = "Hong Kong tourism recovers"
        body = "Hong Kong sees record tourist arrivals as travel restrictions ease across Asia."
        summary = "Elon Musk announced Tesla's new factory in Shanghai will produce advanced robotaxis."
        assert _validate_summary(summary, title, body, "en") is False

    def test_relevant_chinese_summary_passes(self) -> None:
        title = "中国对加拿大实施新关税"
        body = "北京宣布对加拿大进口商品加征关税，涉及农产品和矿产资源。"
        summary = "北京对加拿大商品加征关税，涉及农产品。"
        assert _validate_summary(summary, title, body, "zh") is True

    def test_fabricated_chinese_summary_rejected(self) -> None:
        title = "香港旅游业复苏"
        body = "香港迎来旅游高峰，游客数量创历史新高。"
        summary = "马斯克宣布特斯拉将在上海建设新工厂，生产先进的自动驾驶出租车。"
        assert _validate_summary(summary, title, body, "zh") is False

    def test_empty_source_passes(self) -> None:
        """No source material to compare against — should pass."""
        assert _validate_summary("Any summary", "", "", "en") is True

    def test_very_short_summary_passes(self) -> None:
        """Summaries with fewer than 3 significant words should pass."""
        assert _validate_summary("Hi ok", "Some title", "Some body text here", "en") is True


class TestCleanBilingualFieldFallback:
    """Tests for Fix 3: EN fallback behavior in _clean_bilingual_field."""

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_title_gets_en_fallback_when_zh_empty(self, mock_translate: object) -> None:
        """Titles should still get EN fallback when ZH is empty."""
        mock_translate.return_value = ["", ""]
        signals = [
            {
                "title": {"en": "English title", "zh": ""},
                "body": {"en": "English body", "zh": ""},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        # Title gets EN fallback (blank title is worse)
        assert result[0]["title"]["zh"] == "English title"

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_body_no_en_fallback_when_zh_empty(self, mock_translate: object) -> None:
        """Body should NOT get EN fallback — blank is better than wrong language."""
        mock_translate.return_value = ["中文标题", ""]
        signals = [
            {
                "title": {"en": "English title", "zh": ""},
                "body": {"en": "English body text here", "zh": ""},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        # Body should remain empty, NOT get English text
        assert result[0]["body"]["zh"] == ""

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_perspective_no_en_fallback(self, mock_translate: object) -> None:
        """Perspectives should NOT get EN fallback."""
        mock_translate.return_value = [
            "中文标题", "中文内容",
            "", "",  # Failed perspective translations
        ]
        signals = [
            {
                "title": {"en": "Trade news", "zh": ""},
                "body": {"en": "Trade body", "zh": ""},
                "perspectives": {
                    "canada": {"en": "Ottawa concerned about trade", "zh": ""},
                    "china": {"en": "Beijing justified the move", "zh": ""},
                },
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        # Perspectives should remain empty, NOT get English fallback
        assert result[0]["perspectives"]["canada"]["zh"] == ""
        assert result[0]["perspectives"]["china"]["zh"] == ""

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_zh_all_english_body_cleared(self, mock_translate: object) -> None:
        """ZH body that has only English text should be cleared, not kept."""
        mock_translate.return_value = ["Just English words here"]
        signals = [
            {
                "title": {"en": "Real English title", "zh": "Just English words here"},
                "body": {"en": "Body", "zh": "正文内容"},
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        # Title ZH has no CJK — gets EN fallback (title allowed)
        assert result[0]["title"]["zh"] == "Real English title"

    @patch("analysis.signal_normalization.translate_to_chinese")
    def test_zh_all_english_perspective_cleared(self, mock_translate: object) -> None:
        """ZH perspective with only English should be cleared, not get EN fallback."""
        mock_translate.return_value = [
            "中文标题", "中文内容",
            "Just english no chinese", "Also just english",
        ]
        signals = [
            {
                "title": {"en": "Trade", "zh": ""},
                "body": {"en": "Body", "zh": ""},
                "perspectives": {
                    "canada": {"en": "Ottawa view", "zh": ""},
                    "china": {"en": "Beijing view", "zh": ""},
                },
                "_source_lang": "en",
            }
        ]
        result = translate_signals_batch(signals)
        # Perspectives with all-English ZH should be cleared
        assert result[0]["perspectives"]["canada"]["zh"] == ""
        assert result[0]["perspectives"]["china"]["zh"] == ""
