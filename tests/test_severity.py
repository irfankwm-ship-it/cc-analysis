"""Tests for severity scoring."""

from __future__ import annotations

from datetime import date
from typing import Any

from analysis.classifiers.severity import (
    classify_severity,
    compute_severity_score,
    score_to_severity,
)


class TestScoreToSeverity:
    """Test score-to-severity level mapping."""

    def test_critical(self) -> None:
        # Threshold: >=8 is critical
        assert score_to_severity(8) == "critical"
        assert score_to_severity(10) == "critical"
        assert score_to_severity(15) == "critical"

    def test_high(self) -> None:
        # Threshold: >=6 and <8 is high
        assert score_to_severity(6) == "high"
        assert score_to_severity(7) == "high"

    def test_elevated(self) -> None:
        # Threshold: >=4 and <6 is elevated
        assert score_to_severity(4) == "elevated"
        assert score_to_severity(5) == "elevated"

    def test_moderate(self) -> None:
        # Threshold: >=2 and <4 is moderate
        assert score_to_severity(2) == "moderate"
        assert score_to_severity(3) == "moderate"

    def test_low(self) -> None:
        # Threshold: <2 is low
        assert score_to_severity(0) == "low"
        assert score_to_severity(1) == "low"


class TestComputeSeverityScore:
    """Test raw severity score computation."""

    def test_official_source_adds_4(self) -> None:
        score = compute_severity_score("some text", "official", "diplomatic")
        assert score >= 4

    def test_wire_source_adds_3(self) -> None:
        score = compute_severity_score("some text", "wire", "trade")
        assert score >= 3

    def test_media_source_adds_1(self) -> None:
        score = compute_severity_score("neutral text", "media", "social")
        assert score >= 1

    def test_escalation_keywords_add_points(
        self, severity_modifiers: dict[str, Any]
    ) -> None:
        text = "Detention of citizens amid crisis sanctions"
        score = compute_severity_score(
            text, "wire", "diplomatic", severity_modifiers=severity_modifiers
        )
        # wire(3) + escalation(3) = 6 minimum + bilateral etc.
        assert score >= 6

    def test_deescalation_reduces_score(
        self, severity_modifiers: dict[str, Any]
    ) -> None:
        text_escalation = "Military confrontation crisis"
        text_deescalation = "Agreement cooperation dialogue normalized"

        score_esc = compute_severity_score(
            text_escalation, "media", "military", severity_modifiers=severity_modifiers
        )
        score_deesc = compute_severity_score(
            text_deescalation, "media", "diplomatic", severity_modifiers=severity_modifiers
        )
        assert score_esc > score_deesc

    def test_bilateral_canada_china_adds_2(self) -> None:
        text = "Canada-China relations deteriorating rapidly"
        score = compute_severity_score(text, "media", "diplomatic")
        # media(1) + bilateral(2) = 3 minimum
        assert score >= 3

    def test_general_china_adds_1(self) -> None:
        text = "China announces new policy"
        score = compute_severity_score(text, "media", "political")
        # media(1) + china(1) = 2 minimum
        assert score >= 2

    def test_recency_today_adds_1(self) -> None:
        ref = date(2025, 1, 30)
        score = compute_severity_score(
            "some text", "media", "trade",
            signal_date="2025-01-30",
            reference_date=ref,
        )
        # media(1) + recency(1) = 2 minimum
        assert score >= 2

    def test_recency_old_subtracts(self) -> None:
        ref = date(2025, 1, 30)
        score_today = compute_severity_score(
            "test text", "media", "trade",
            signal_date="2025-01-30",
            reference_date=ref,
        )
        score_old = compute_severity_score(
            "test text", "media", "trade",
            signal_date="2025-01-01",
            reference_date=ref,
        )
        assert score_today > score_old

    def test_score_never_negative(self, severity_modifiers: dict[str, Any]) -> None:
        text = "Agreement cooperation dialogue resolved eased"
        score = compute_severity_score(
            text, "media", "diplomatic", severity_modifiers=severity_modifiers
        )
        assert score >= 0


class TestClassifySeverity:
    """Test signal-level severity classification."""

    def test_high_severity_diplomatic(
        self,
        sample_diplomatic_signal: dict[str, Any],
        severity_modifiers: dict[str, Any],
    ) -> None:
        result = classify_severity(
            sample_diplomatic_signal,
            source_tier="official",
            category="diplomatic",
            severity_modifiers=severity_modifiers,
        )
        # official(4) + bilateral(2) + moderate_escalation("tension")(2) = 8 -> high
        assert result in ("high", "critical", "elevated")

    def test_trade_signal_with_sanctions(
        self,
        sample_trade_signal: dict[str, Any],
        severity_modifiers: dict[str, Any],
    ) -> None:
        result = classify_severity(
            sample_trade_signal,
            source_tier="official",
            category="trade",
            severity_modifiers=severity_modifiers,
        )
        # Contains "sanctions" -> escalation weight
        assert result in ("high", "critical")

    def test_low_severity_neutral_text(
        self, severity_modifiers: dict[str, Any]
    ) -> None:
        signal = {"title": "General report", "body": "Nothing significant"}
        result = classify_severity(
            signal,
            source_tier="media",
            category="social",
            severity_modifiers=severity_modifiers,
        )
        assert result in ("low", "moderate")
