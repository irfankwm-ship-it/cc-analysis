"""Tests for active situation tracking."""

from __future__ import annotations

from analysis.active_situations import ActiveSituation, track_situations


class TestTrackSituations:
    """Test situation tracking based on signal patterns."""

    def test_canola_situation_detected(self) -> None:
        """Signals mentioning canola should trigger canola trade dispute."""
        signals = [
            {
                "title": {
                    "en": "New canola tariff",
                    "zh": "\u65B0\u6CB9\u83DC\u7C7D\u5173\u7A0E",
                },
                "body": {
                    "en": "China restricts canola imports",
                    "zh": "\u4E2D\u56FD\u9650\u5236\u6CB9\u83DC\u7C7D\u8FDB\u53E3",
                },
                "category": "trade",
                "severity": "high",
            }
        ]
        result = track_situations(signals, "2025-01-30")

        names = [s.name["en"] for s in result]
        assert "Canola Trade Dispute" in names

    def test_tech_decoupling_detected(self) -> None:
        """Signals about Huawei/5G should trigger tech decoupling."""
        signals = [
            {
                "title": {
                    "en": "Huawei 5G ban extended",
                    "zh": "\u534E\u4E3A5G\u7981\u4EE4\u5EF6\u957F",
                },
                "body": {
                    "en": "Government extends ban",
                    "zh": "\u653F\u5E9C\u5EF6\u957F\u7981\u4EE4",
                },
                "category": "technology",
                "severity": "high",
            }
        ]
        result = track_situations(signals, "2025-01-30")

        names = [s.name["en"] for s in result]
        assert "Tech Decoupling" in names

    def test_foreign_interference_detected(self) -> None:
        """Signals about foreign interference should trigger situation."""
        signals = [
            {
                "title": "Foreign interference inquiry continues",
                "body": "CSIS presents evidence of interference",
                "category": "political",
                "severity": "elevated",
            }
        ]
        result = track_situations(signals, "2025-01-30")

        names = [s.name["en"] for s in result]
        assert "Foreign Interference Investigation" in names

    def test_no_matching_signals(self) -> None:
        """Signals with no matching keywords should produce no situations."""
        signals = [
            {
                "title": "Weather forecast update",
                "body": "Sunny skies expected tomorrow",
                "category": "social",
                "severity": "low",
            }
        ]
        result = track_situations(signals, "2025-01-30")
        assert len(result) == 0

    def test_day_count_computation(self) -> None:
        """Day count should reflect time since situation start date."""
        signals = [
            {
                "title": "Canola tariff news",
                "body": "Canola trade update",
                "category": "trade",
                "severity": "moderate",
            }
        ]
        result = track_situations(signals, "2025-01-30")

        canola = next((s for s in result if s.name["en"] == "Canola Trade Dispute"), None)
        assert canola is not None
        # Start date 2019-03-01, current 2025-01-30 = ~2161 days
        assert canola.day_count > 2000

    def test_severity_upgrade(self) -> None:
        """Situation severity should upgrade based on signal severity."""
        signals = [
            {
                "title": "Taiwan Strait military exercise",
                "body": "PLA conducts exercise near Taiwan",
                "category": "military",
                "severity": "critical",
            }
        ]
        result = track_situations(signals, "2025-01-30")

        taiwan = next((s for s in result if s.name["en"] == "Taiwan Strait Tensions"), None)
        assert taiwan is not None
        # Default is "elevated" but signal is "critical", so should upgrade
        assert taiwan.severity == "critical"

    def test_sorted_by_severity(self) -> None:
        """Results should be sorted by severity (highest first)."""
        signals = [
            {
                "title": "Canola dispute continues",
                "body": "Trade concerns over canola",
                "category": "trade",
                "severity": "moderate",
            },
            {
                "title": "Foreign interference probe",
                "body": "CSIS investigation ongoing",
                "category": "political",
                "severity": "high",
            },
        ]
        result = track_situations(signals, "2025-01-30")

        if len(result) >= 2:
            severity_order = {"critical": 5, "high": 4, "elevated": 3, "moderate": 2, "low": 1}
            for i in range(len(result) - 1):
                assert severity_order[result[i].severity] >= severity_order[result[i + 1].severity]

    def test_empty_signals(self) -> None:
        """Empty signal list should produce no situations."""
        result = track_situations([], "2025-01-30")
        assert len(result) == 0

    def test_multiple_situations_from_one_signal(self) -> None:
        """One signal can trigger multiple situations."""
        signals = [
            {
                "title": "Huawei semiconductor rare earth restrictions",
                "body": "Gallium germanium export controls and Huawei 5G ban",
                "category": "technology",
                "severity": "high",
            }
        ]
        result = track_situations(signals, "2025-01-30")

        names = [s.name["en"] for s in result]
        assert "Tech Decoupling" in names
        assert "Rare Earth Export Controls" in names


class TestActiveSituationSerialization:
    """Test ActiveSituation serialization."""

    def test_to_dict(self) -> None:
        situation = ActiveSituation(
            name={"en": "Test Situation", "zh": "\u6D4B\u8BD5\u60C5\u51B5"},
            detail={"en": "Details here", "zh": "\u8BE6\u60C5\u5728\u6B64"},
            severity="elevated",
            day_count=100,
        )
        d = situation.to_dict()
        assert d["name"]["en"] == "Test Situation"
        assert d["severity"] == "elevated"
        assert d["day_count"] == 100

    def test_to_dict_without_optional_fields(self) -> None:
        situation = ActiveSituation(
            name={"en": "Test", "zh": "\u6D4B\u8BD5"},
            detail={"en": "Detail", "zh": "\u8BE6\u60C5"},
            severity="low",
        )
        d = situation.to_dict()
        assert "day_count" not in d
        assert "deadline" not in d
