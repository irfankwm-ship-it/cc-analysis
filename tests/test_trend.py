"""Tests for day-over-day trend computation."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.trend import compute_trends


class TestComputeTrends:
    """Test trend computation logic."""

    def test_no_previous_returns_empty_trends(self) -> None:
        """When no previous briefing exists, all trends should be empty."""
        result = compute_trends(
            current_date="2025-01-30",
            current_signals=[{"category": "trade", "severity": "high"}],
            processed_dir="/nonexistent/path",
            archive_dir="/nonexistent/path",
        )
        assert result.has_previous is False
        assert result.previous_composite is None
        assert result.previous_components == {}
        assert result.previous_signal_count == 0

    def test_with_previous_briefing(self, tmp_path: Path) -> None:
        """Test trend computation when previous briefing exists."""
        # Create a previous day's briefing
        prev_date = "2025-01-29"
        prev_dir = tmp_path / "processed" / prev_date
        prev_dir.mkdir(parents=True)

        prev_briefing = {
            "date": prev_date,
            "signals": [
                {"category": "trade", "severity": "high"},
                {"category": "diplomatic", "severity": "elevated"},
            ],
            "tension_index": {
                "composite": 5.0,
                "components": [
                    {"name": {"en": "Trade", "zh": "\u8D38\u6613"}, "score": 6},
                    {"name": {"en": "Diplomatic", "zh": "\u5916\u4EA4"}, "score": 4},
                ],
            },
        }
        with open(prev_dir / "briefing.json", "w") as f:
            json.dump(prev_briefing, f)

        current_signals = [
            {"category": "trade", "severity": "critical"},
            {"category": "trade", "severity": "high"},
            {"category": "military", "severity": "elevated"},
        ]

        result = compute_trends(
            current_date="2025-01-30",
            current_signals=current_signals,
            processed_dir=str(tmp_path / "processed"),
            archive_dir=str(tmp_path / "archive"),
        )

        assert result.has_previous is True
        assert result.previous_composite == 5.0
        assert result.previous_components["trade"] == 6
        assert result.previous_components["diplomatic"] == 4
        assert result.previous_signal_count == 2
        assert result.new_signals_delta == 1

    def test_category_shifts(self, tmp_path: Path) -> None:
        """Test category shift detection."""
        prev_dir = tmp_path / "processed" / "2025-01-29"
        prev_dir.mkdir(parents=True)

        prev_briefing = {
            "signals": [
                {"category": "trade", "severity": "high"},
                {"category": "trade", "severity": "moderate"},
            ],
            "tension_index": {"composite": 3.0, "components": []},
        }
        with open(prev_dir / "briefing.json", "w") as f:
            json.dump(prev_briefing, f)

        current_signals = [
            {"category": "trade", "severity": "high"},
            {"category": "diplomatic", "severity": "elevated"},
        ]

        result = compute_trends(
            current_date="2025-01-30",
            current_signals=current_signals,
            processed_dir=str(tmp_path / "processed"),
            archive_dir="",
        )

        assert result.category_shifts["trade"] == "down"  # 2 -> 1
        assert result.category_shifts["diplomatic"] == "up"  # 0 -> 1

    def test_archive_fallback(self, tmp_path: Path) -> None:
        """Test that archive directory is searched when processed dir fails."""
        archive_dir = tmp_path / "archive" / "daily" / "2025-01-29"
        archive_dir.mkdir(parents=True)

        prev_briefing = {
            "signals": [{"category": "military", "severity": "low"}],
            "tension_index": {"composite": 1.0, "components": []},
        }
        with open(archive_dir / "briefing.json", "w") as f:
            json.dump(prev_briefing, f)

        result = compute_trends(
            current_date="2025-01-30",
            current_signals=[],
            processed_dir="/nonexistent",
            archive_dir=str(tmp_path / "archive"),
        )

        assert result.has_previous is True
        assert result.previous_composite == 1.0

    def test_corrupted_previous_file(self, tmp_path: Path) -> None:
        """Gracefully handle corrupted previous briefing."""
        prev_dir = tmp_path / "processed" / "2025-01-29"
        prev_dir.mkdir(parents=True)
        with open(prev_dir / "briefing.json", "w") as f:
            f.write("not valid json{{{")

        result = compute_trends(
            current_date="2025-01-30",
            current_signals=[],
            processed_dir=str(tmp_path / "processed"),
            archive_dir="",
        )
        assert result.has_previous is False
