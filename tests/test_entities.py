"""Tests for entity matching."""

from __future__ import annotations

from analysis.entities import (
    build_entity_directory,
    match_entities_across_signals,
    match_entities_in_signal,
)


class TestMatchEntitiesInSignal:
    """Test entity matching against individual signals."""

    def test_english_match(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": {"en": "Xi Jinping meets with delegation", "zh": ""},
            "body": {"en": "President Xi discussed trade with MOFCOM officials", "zh": ""},
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "xi_jinping" in result
        assert "mofcom" in result

    def test_chinese_match(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        zh_title = "\u4E60\u8FD1\u5E73\u4F1A\u89C1\u5916\u56FD\u4EE3\u8868\u56E2"
        signal = {
            "title": {"en": "", "zh": zh_title},
            "body": {"en": "", "zh": "\u5546\u52A1\u90E8\u53D1\u8868\u58F0\u660E"},
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "xi_jinping" in result
        assert "mofcom" in result

    def test_canola_match(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": "New tariff on canola seed imports",
            "body": "Oilseed exports face restrictions",
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "canola" in result

    def test_huawei_match(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": "Huawei 5G ban extended",
            "body": "The ban on Huawei equipment continues",
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "huawei" in result

    def test_rare_earths_match(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": "China restricts rare earth exports",
            "body": "Gallium and germanium export controls tighten",
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "rare_earths" in result

    def test_two_michaels_match(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": "Michael Kovrig and Michael Spavor case update",
            "body": "The Two Michaels saga continues",
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "two_michaels" in result

    def test_no_match(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": "Weather report for today",
            "body": "Sunny skies expected",
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert len(result) == 0

    def test_multiple_entities(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": "Xi Jinping discusses Huawei ban with Wang Yi",
            "body": "MFA responds to canola tariff concerns. CSIS reports on rare earth strategy.",
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "xi_jinping" in result
        assert "huawei" in result
        assert "wang_yi" in result
        assert "mfa" in result
        assert "canola" in result
        assert "csis" in result
        assert "rare_earths" in result

    def test_implications_scanned(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signal = {
            "title": "Trade update",
            "body": "General news",
            "implications": {
                "canada_impact": {
                    "en": "Canola farmers affected",
                    "zh": "\u6CB9\u83DC\u7C7D\u519C\u6C11\u53D7\u5F71\u54CD",
                },
            },
        }
        result = match_entities_in_signal(signal, entity_aliases)
        assert "canola" in result


class TestMatchEntitiesAcrossSignals:
    """Test entity matching across multiple signals."""

    def test_aggregation(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signals = [
            {"title": "Xi Jinping speech", "body": "MOFCOM announcement"},
            {"title": "Xi Jinping meets delegation", "body": "Canola tariff news"},
            {"title": "Weather update", "body": "No entities here"},
        ]
        result = match_entities_across_signals(signals, entity_aliases)

        # xi_jinping mentioned in 2 signals
        xi_match = next((m for m in result if m.entity_id == "xi_jinping"), None)
        assert xi_match is not None
        assert xi_match.mention_count == 2

    def test_sorted_by_mentions(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        signals = [
            {"title": "Huawei", "body": ""},
            {"title": "Huawei ban", "body": ""},
            {"title": "Huawei 5G", "body": ""},
            {"title": "Canola tariff", "body": ""},
        ]
        result = match_entities_across_signals(signals, entity_aliases)
        if len(result) >= 2:
            assert result[0].mention_count >= result[1].mention_count

    def test_empty_signals(self, entity_aliases: dict[str, dict[str, list[str]]]) -> None:
        result = match_entities_across_signals([], entity_aliases)
        assert len(result) == 0


class TestBuildEntityDirectory:
    """Test entity directory building."""

    def test_builds_correct_structure(
        self, entity_aliases: dict[str, dict[str, list[str]]]
    ) -> None:
        from analysis.entities import EntityMatch

        matches = [
            EntityMatch(entity_id="xi_jinping", mention_count=3),
            EntityMatch(entity_id="canola", mention_count=2),
        ]
        result = build_entity_directory(matches, entity_aliases)

        assert len(result) == 2
        assert result[0]["id"] == "xi_jinping"
        assert result[0]["name"]["en"] == "Xi Jinping"
        assert result[0]["type"] == "people"
        assert result[0]["has_detail_page"] is False

        assert result[1]["id"] == "canola"
        assert result[1]["name"]["en"] == "canola"
        assert result[1]["type"] == "commodity"
