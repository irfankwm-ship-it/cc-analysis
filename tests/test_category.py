"""Tests for category classification."""

from __future__ import annotations

from typing import Any

from analysis.classifiers.category import classify_category, classify_signal


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


class TestFallbackCategory:
    """Test _fallback_category for crime keywords."""

    def test_crime_fallback(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        """Crime text without keyword dict matches should fall back to legal."""
        from analysis.classifiers.category import _fallback_category
        assert _fallback_category("Man arrested for fraud in money laundering scheme") == "legal"

    def test_mule_fallback(self, categories_dict: dict[str, dict[str, list[str]]]) -> None:
        from analysis.classifiers.category import _fallback_category
        assert _fallback_category("Suspect convicted as mule in criminal trial") == "legal"
