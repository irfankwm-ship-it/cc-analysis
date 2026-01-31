"""Tests for monthly volume compilation."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.volume_compiler import compile_volume, write_volume


class TestCompileVolume:
    """Test volume compilation from daily briefings."""

    def test_empty_archive(self, tmp_path: Path) -> None:
        """Volume with no daily briefings should have zero signals."""
        result = compile_volume("2025-02-01", str(tmp_path))
        assert result["signal_count"] == 0
        assert result["volume_number"] == 1
        assert result["category_breakdown"] == {}
        assert result["severity_breakdown"] == {}

    def test_aggregation(self, tmp_path: Path) -> None:
        """Test aggregation of daily briefings into a volume."""
        daily_dir = tmp_path / "daily"

        # Create 3 days of briefings for January 2025
        for day in ["2025-01-28", "2025-01-29", "2025-01-30"]:
            day_dir = daily_dir / day
            day_dir.mkdir(parents=True)
            briefing = {
                "date": day,
                "signals": [
                    {"category": "trade", "severity": "high"},
                    {"category": "diplomatic", "severity": "elevated"},
                ],
                "tension_index": {"composite": 5.5},
            }
            with open(day_dir / "briefing.json", "w") as f:
                json.dump(briefing, f)

        # Compile for February 1 (compiles January)
        result = compile_volume("2025-02-01", str(tmp_path))

        assert result["signal_count"] == 6  # 3 days * 2 signals
        assert result["category_breakdown"]["trade"] == 3
        assert result["category_breakdown"]["diplomatic"] == 3
        assert result["severity_breakdown"]["high"] == 3
        assert result["severity_breakdown"]["elevated"] == 3
        assert result["period_start"] == "2025-01-01"
        assert result["period_end"] == "2025-01-31"

    def test_tension_trend(self, tmp_path: Path) -> None:
        """Test tension trend line extraction."""
        daily_dir = tmp_path / "daily"

        for day, composite in [("2025-01-28", 4.0), ("2025-01-29", 5.5), ("2025-01-30", 6.2)]:
            day_dir = daily_dir / day
            day_dir.mkdir(parents=True)
            briefing = {
                "date": day,
                "signals": [],
                "tension_index": {"composite": composite},
            }
            with open(day_dir / "briefing.json", "w") as f:
                json.dump(briefing, f)

        result = compile_volume("2025-02-01", str(tmp_path))
        assert len(result["tension_trend"]) == 3
        assert result["tension_trend"][0]["value"] == 4.0
        assert result["tension_trend"][2]["value"] == 6.2

    def test_volume_numbering(self, tmp_path: Path) -> None:
        """Test volume number auto-increment."""
        volumes_dir = tmp_path / "volumes"
        volumes_dir.mkdir(parents=True)

        # Create existing volume
        with open(volumes_dir / "vol-001.json", "w") as f:
            json.dump({"volume_number": 1}, f)

        result = compile_volume("2025-02-01", str(tmp_path))
        assert result["volume_number"] == 2


class TestWriteVolume:
    """Test volume file writing."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        volume_meta = {
            "volume_number": 1,
            "period_start": "2025-01-01",
            "period_end": "2025-01-31",
            "signal_count": 42,
        }

        path = write_volume(volume_meta, str(tmp_path))
        assert path.exists()
        assert path.name == "vol-001.json"

        with open(path) as f:
            data = json.load(f)
        assert data["volume_number"] == 1
        assert data["signal_count"] == 42
