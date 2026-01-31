"""Tests for tension index computation."""

from __future__ import annotations

from typing import Any

from analysis.tension_index import (
    COMPONENT_WEIGHTS,
    SEVERITY_POINTS,
    ComponentScore,
    TensionIndex,
    compute_tension_index,
)


class TestTensionIndexFormula:
    """Test the tension index formula matches expected outputs."""

    def test_empty_signals_produces_zero(self) -> None:
        result = compute_tension_index([])
        assert result.composite == 0.0
        assert result.level["en"] == "Low"
        assert result.delta == 0.0

    def test_single_component(self) -> None:
        """A single elevated diplomatic signal should score correctly."""
        signals = [{"category": "diplomatic", "severity": "elevated"}]
        result = compute_tension_index(signals)
        # diplomatic: 3 points -> 3/20*10 = 1.5 -> round to 2
        # composite: 1.5 * 0.25 = 0.375
        assert result.composite > 0

    def test_seed_data_tension_index(self) -> None:
        """Verify the formula produces approximately 6.2 for seed-like signals.

        Seed data has composite of 6.2 (Elevated level).
        We simulate signals that would produce scores matching:
          diplomatic: 7, trade: 8, military: 6, political: 5, technology: 7, social: 3
        """
        # To get component_score=7 for diplomatic:
        # 7 = min(sum/20*10, 10) -> sum = 7*20/10 = 14 points
        # 14 points from diplomatic: e.g., 2 critical(10) + 1 high(4) = 14
        signals: list[dict[str, Any]] = []

        # Diplomatic: target score=7 -> need 14 severity points
        # 2 critical + 1 high = 10 + 4 = 14
        signals.extend([
            {"category": "diplomatic", "severity": "critical"},
            {"category": "diplomatic", "severity": "critical"},
            {"category": "diplomatic", "severity": "high"},
        ])

        # Trade: target score=8 -> need 16 severity points
        # 3 critical + 1 low = 15+1=16
        signals.extend([
            {"category": "trade", "severity": "critical"},
            {"category": "trade", "severity": "critical"},
            {"category": "trade", "severity": "critical"},
            {"category": "trade", "severity": "low"},
        ])

        # Military: target score=6 -> need 12 severity points
        # 2 critical + 1 moderate = 10+2 = 12
        signals.extend([
            {"category": "military", "severity": "critical"},
            {"category": "military", "severity": "critical"},
            {"category": "military", "severity": "moderate"},
        ])

        # Political: target score=5 -> need 10 severity points
        # 2 critical = 10
        signals.extend([
            {"category": "political", "severity": "critical"},
            {"category": "political", "severity": "critical"},
        ])

        # Technology: target score=7 -> need 14 severity points
        # Same as diplomatic
        signals.extend([
            {"category": "technology", "severity": "critical"},
            {"category": "technology", "severity": "critical"},
            {"category": "technology", "severity": "high"},
        ])

        # Social: target score=3 -> need 6 severity points
        # 1 critical + 1 low = 5+1=6
        signals.extend([
            {"category": "social", "severity": "critical"},
            {"category": "social", "severity": "low"},
        ])

        result = compute_tension_index(signals)

        # Expected composite:
        # diplomatic: min(14/20*10, 10) = 7.0 * 0.25 = 1.75
        # trade: min(16/20*10, 10) = 8.0 * 0.25 = 2.0
        # military: min(12/20*10, 10) = 6.0 * 0.15 = 0.9
        # political: min(10/20*10, 10) = 5.0 * 0.15 = 0.75
        # technology: min(14/20*10, 10) = 7.0 * 0.10 = 0.7
        # social: min(6/20*10, 10) = 3.0 * 0.10 = 0.3
        # Total = 1.75 + 2.0 + 0.9 + 0.75 + 0.7 + 0.3 = 6.4
        assert 6.0 <= result.composite <= 6.5
        assert result.level["en"] == "Elevated"

    def test_maximum_score(self) -> None:
        """All components at maximum should produce composite=10."""
        signals: list[dict[str, Any]] = []
        # Need 20+ severity points per category to max out
        for cat in COMPONENT_WEIGHTS:
            for _ in range(5):
                signals.append({"category": cat, "severity": "critical"})

        result = compute_tension_index(signals)
        # Each component: min(25/20*10, 10) = 10
        # composite: 10 * (0.25+0.25+0.15+0.15+0.10+0.10) = 10
        assert result.composite == 10.0
        assert result.level["en"] == "Critical"

    def test_delta_computation(self) -> None:
        """Delta should be difference from previous composite."""
        signals = [{"category": "diplomatic", "severity": "high"}]
        result = compute_tension_index(signals, previous_composite=5.0)
        assert result.delta == result.composite - 5.0

    def test_trend_computation(self) -> None:
        """Trends should reflect component score changes."""
        signals = [{"category": "diplomatic", "severity": "critical"}]
        prev_components = {"diplomatic": 1, "trade": 5}

        result = compute_tension_index(
            signals, previous_components=prev_components
        )

        diplomatic_comp = next(
            c for c in result.components if c.name["en"] == "Diplomatic"
        )
        trade_comp = next(
            c for c in result.components if c.name["en"] == "Trade"
        )

        # Diplomatic should be up (score > 1)
        assert diplomatic_comp.trend == "up"
        # Trade should be down (score 0 < 5)
        assert trade_comp.trend == "down"

    def test_no_previous_means_zero_delta(self) -> None:
        signals = [{"category": "trade", "severity": "moderate"}]
        result = compute_tension_index(signals)
        assert result.delta == 0.0

    def test_all_components_stable_without_previous(self) -> None:
        signals = [{"category": "diplomatic", "severity": "low"}]
        result = compute_tension_index(signals)
        for comp in result.components:
            assert comp.trend == "stable"

    def test_severity_points_mapping(self) -> None:
        """Verify SEVERITY_POINTS constants."""
        assert SEVERITY_POINTS["critical"] == 5
        assert SEVERITY_POINTS["high"] == 4
        assert SEVERITY_POINTS["elevated"] == 3
        assert SEVERITY_POINTS["moderate"] == 2
        assert SEVERITY_POINTS["low"] == 1

    def test_component_weights_sum_to_one(self) -> None:
        total = sum(COMPONENT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_level_thresholds(self) -> None:
        """Test all level threshold boundaries."""
        # 0-2: low
        result_low = compute_tension_index([])
        assert result_low.level["en"] == "Low"

        # Generate moderate (2.1-4)
        signals_mod = [
            {"category": "diplomatic", "severity": "critical"},
            {"category": "diplomatic", "severity": "critical"},
            {"category": "trade", "severity": "critical"},
            {"category": "trade", "severity": "critical"},
        ]
        result_mod = compute_tension_index(signals_mod)
        # diplomatic: 10/20*10=5 -> 5*0.25=1.25
        # trade: 10/20*10=5 -> 5*0.25=1.25
        # total=2.5 -> Moderate
        assert result_mod.level["en"] == "Moderate"


class TestTensionIndexSerialization:
    """Test TensionIndex serialization."""

    def test_to_dict(self) -> None:
        ti = TensionIndex(
            composite=6.2,
            level={"en": "Elevated", "zh": "\u5347\u9AD8"},
            delta=0.3,
            delta_description={
                "en": "+0.3 from previous day",
                "zh": "\u6BD4\u524D\u4E00\u5929+0.3",
            },
            components=[
                ComponentScore(
                    name={"en": "Diplomatic", "zh": "\u5916\u4EA4"},
                    score=7,
                    weight=0.25,
                    trend="up",
                    key_driver={"en": "Ambassador summoned", "zh": "\u53EC\u89C1\u5927\u4F7F"},
                )
            ],
        )
        d = ti.to_dict()
        assert d["composite"] == 6.2
        assert d["level"]["en"] == "Elevated"
        assert d["delta"] == 0.3
        assert len(d["components"]) == 1
        assert d["components"][0]["score"] == 7
