"""Tests for category classification."""

from __future__ import annotations

from typing import Any

from analysis.classifiers.category import classify_category, classify_signal, validate_category


class TestClassifyCategory:
    """Test keyword-based category classification."""

    def test_diplomatic_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "The ambassador was summoned to discuss bilateral diplomatic relations"
        assert classify_category(text, categories_dict) == "diplomatic"

    def test_trade_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "New tariff imposed on imports, WTO investigates trade deficit dumping claims"
        assert classify_category(text, categories_dict) == "trade"

    def test_military_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "PLA navy conducts military exercise with warship deployment in Taiwan Strait"
        assert classify_category(text, categories_dict) == "military"

    def test_technology_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "Huawei 5G semiconductor chip ban restricts rare earth gallium germanium"
        assert classify_category(text, categories_dict) == "technology"

    def test_political_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "Parliament legislation on foreign interference bill targets CCP election influence"
        assert classify_category(text, categories_dict) == "political"

    def test_economic_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = (
            "PBOC cuts reserve requirement and interest rate"
            " to stimulate GDP growth, yuan weakens"
        )
        assert classify_category(text, categories_dict) == "economic"

    def test_social_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "University student diaspora surveillance Confucius Institute cultural education"
        assert classify_category(text, categories_dict) == "social"

    def test_legal_text(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "Extradition court judicial prosecution antitrust regulation SAMR compliance"
        assert classify_category(text, categories_dict) == "legal"

    def test_empty_text_defaults_to_political(
        self, categories_dict: dict[str, dict[str, list[str]]]
    ) -> None:
        assert classify_category("", categories_dict) == "political"

    def test_chinese_diplomatic(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "\u5927\u4F7F\u88AB\u53EC\u89C1\u8BA8\u8BBA\u53CC\u8FB9\u5916\u4EA4\u5173\u7CFB"
        assert classify_category(text, categories_dict) == "diplomatic"

    def test_chinese_trade(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "\u5173\u7A0E\u8D38\u6613\u51FA\u53E3\u8FDB\u53E3\u5236\u88C1\u7981\u8FD0"
        assert classify_category(text, categories_dict) == "trade"

    def test_chinese_military(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = (
            "\u89E3\u653E\u519B\u6D77\u519B\u5728\u53F0\u6E7E"
            "\u6D77\u5CE1\u8FDB\u884C\u519B\u4E8B\u6F14\u4E60"
        )
        assert classify_category(text, categories_dict) == "military"


class TestClassifySignal:
    """Test signal-level classification."""

    def test_bilingual_signal(
        self,
        sample_diplomatic_signal: dict[str, Any],
        categories_dict: dict[str, dict[str, list[str]]],
    ) -> None:
        result = classify_signal(sample_diplomatic_signal, categories_dict)
        assert result == "diplomatic"

    def test_trade_signal(
        self,
        sample_trade_signal: dict[str, Any],
        categories_dict: dict[str, dict[str, list[str]]],
    ) -> None:
        result = classify_signal(sample_trade_signal, categories_dict)
        assert result == "trade"

    def test_military_signal(
        self,
        sample_military_signal: dict[str, Any],
        categories_dict: dict[str, dict[str, list[str]]],
    ) -> None:
        result = classify_signal(sample_military_signal, categories_dict)
        assert result == "military"

    def test_tech_signal(
        self,
        sample_tech_signal: dict[str, Any],
        categories_dict: dict[str, dict[str, list[str]]],
    ) -> None:
        result = classify_signal(sample_tech_signal, categories_dict)
        assert result == "technology"

    def test_political_signal(
        self,
        sample_political_signal: dict[str, Any],
        categories_dict: dict[str, dict[str, list[str]]],
    ) -> None:
        result = classify_signal(sample_political_signal, categories_dict)
        assert result == "political"

    def test_social_signal(
        self,
        sample_social_signal: dict[str, Any],
        categories_dict: dict[str, dict[str, list[str]]],
    ) -> None:
        result = classify_signal(sample_social_signal, categories_dict)
        assert result == "social"

    def test_string_title_signal(
        self, categories_dict: dict[str, dict[str, list[str]]]
    ) -> None:
        signal = {"title": "PLA military exercise Taiwan Strait", "body": "Navy warship"}
        result = classify_signal(signal, categories_dict)
        assert result == "military"

    def test_headline_field(
        self, categories_dict: dict[str, dict[str, list[str]]]
    ) -> None:
        signal = {"headline": "New tariff on trade imports increases trade deficit"}
        result = classify_signal(signal, categories_dict)
        assert result == "trade"

    def test_empty_signal(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        result = classify_signal({}, categories_dict)
        assert result == "political"


class TestTradeKeywords:
    """Test automotive/EV keywords classify as trade."""

    def test_byd_auto(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "BYD expands electric vehicle factory production in Southeast Asia"
        assert classify_category(text, categories_dict) == "trade"

    def test_ev_exports(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "China EV automaker NIO announces new vehicle exports to Canada"
        assert classify_category(text, categories_dict) == "trade"

    def test_catl_battery(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "CATL automotive battery production ramps up for Geely"
        assert classify_category(text, categories_dict) == "trade"

    def test_chinese_ev_keywords(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "比亚迪新能源汽车出口大幅增长，宁德时代电池产能扩张"
        assert classify_category(text, categories_dict) == "trade"


class TestLegalKeywords:
    """Test crime/enforcement keywords classify as legal."""

    def test_money_mule(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "Chinese national arrested as money mule in fraud laundering scheme"
        assert classify_category(text, categories_dict) == "legal"

    def test_smuggling(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "Court sentences smuggling ring trafficking goods from China"
        assert classify_category(text, categories_dict) == "legal"

    def test_money_laundering(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "China money laundering criminal convicted prison sentence verdict"
        assert classify_category(text, categories_dict) == "legal"

    def test_chinese_legal_keywords(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "走私集团被逮捕，洗钱犯罪嫌疑人在审判中被判有罪"
        assert classify_category(text, categories_dict) == "legal"


class TestMilitaryKeywords:
    """Test military keyword additions (submarine, fleet, etc.)."""

    def test_nuclear_submarine_fleet(
        self, categories_dict: dict[str, dict[str, list[str]]]
    ) -> None:
        text = "China launches nuclear submarine fleet in South China Sea"
        assert classify_category(text, categories_dict) == "military"

    def test_carrier_destroyer(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "PLA Navy carrier and destroyer group conducts exercises"
        assert classify_category(text, categories_dict) == "military"

    def test_ballistic_missile(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "China tests ballistic missile submarine capability"
        assert classify_category(text, categories_dict) == "military"


class TestDiplomaticKeywords:
    """Test diplomatic keyword additions (condemn, sovereignty, etc.)."""

    def test_condemns_lai(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "China condemns Lai war instigator remarks at press conference"
        assert classify_category(text, categories_dict) == "diplomatic"

    def test_denounces_sovereignty(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "Beijing denounces violations of sovereignty and territorial integrity"
        assert classify_category(text, categories_dict) == "diplomatic"

    def test_slams_rebukes(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        text = "China slams and rebukes foreign interference in domestic affairs"
        assert classify_category(text, categories_dict) == "diplomatic"


class TestAISubstringBug:
    """Test that 'AI' as a single-word keyword doesn't match substrings like 'lai'."""

    def test_ai_semiconductor_still_technology(
        self, categories_dict: dict[str, dict[str, list[str]]]
    ) -> None:
        text = "AI semiconductor chip development accelerates"
        assert classify_category(text, categories_dict) == "technology"

    def test_lai_not_technology(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        """'Lai Ching-te' should NOT match 'AI' keyword via substring."""
        text = "Lai Ching-te interview on cross-strait relations"
        result = classify_category(text, categories_dict)
        assert result != "technology"


class TestFallbackCategory:
    """Test _fallback_category for crime keywords."""

    def test_crime_fallback(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        """Crime text without keyword dict matches should fall back to legal."""
        from analysis.classifiers.category import _fallback_category
        assert _fallback_category("Man arrested for fraud in money laundering scheme") == "legal"

    def test_mule_fallback(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        from analysis.classifiers.category import _fallback_category
        assert _fallback_category("Suspect convicted as mule in criminal trial") == "legal"


class TestValidateCategory:
    """Tests for validate_category override rules."""

    def test_overrides_technology_to_trade_for_tariff(self) -> None:
        text = "China imposes new tariff on imports, export restrictions tightened"
        result = validate_category(text, "technology")
        assert result == "trade"

    def test_overrides_technology_to_diplomatic_for_embassy(self) -> None:
        text = "Ambassador summoned to embassy for diplomatic discussions"
        result = validate_category(text, "technology")
        assert result == "diplomatic"

    def test_overrides_technology_to_military(self) -> None:
        text = "Coast guard navy deployed near contested waters"
        result = validate_category(text, "technology")
        assert result == "military"

    def test_does_not_override_correct_trade(self) -> None:
        text = "New tariff on imports increases trade deficit"
        result = validate_category(text, "trade")
        assert result == "trade"

    def test_does_not_override_correct_diplomatic(self) -> None:
        text = "Ambassador meets with foreign ministry officials"
        result = validate_category(text, "diplomatic")
        assert result == "diplomatic"

    def test_requires_two_indicators(self) -> None:
        """Single indicator is not enough to override."""
        text = "New tariff announced on Chinese goods"
        result = validate_category(text, "technology")
        assert result == "technology"

    def test_trade_override_with_chinese_text(self) -> None:
        text = "关税进口贸易争端 technology chip semiconductor"
        result = validate_category(text, "technology")
        assert result == "trade"

    def test_diplomatic_does_not_override_trade(self) -> None:
        """Diplomatic indicators should not override trade category."""
        text = "Ambassador discusses trade tariffs at embassy"
        result = validate_category(text, "trade")
        assert result == "trade"

    def test_military_does_not_override_diplomatic(self) -> None:
        """Military indicators should not override diplomatic category."""
        text = "Navy deployed near embassy area, missile concerns"
        result = validate_category(text, "diplomatic")
        assert result == "diplomatic"

    def test_xie_feng_diplomatic_override(self) -> None:
        """The Xie Feng signal should be overridden from technology to diplomatic."""
        text = (
            "Xie Feng states that there are differences and contradictions "
            "between China and the US; the key is to respect each other's core "
            "interests and major concerns. Ambassador diplomatic foreign ministry"
        )
        result = validate_category(text, "technology")
        assert result == "diplomatic"
