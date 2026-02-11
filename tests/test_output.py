"""Tests for output module."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.output import (
    assemble_briefing,
    validate_briefing,
    write_archive,
    write_processed,
)


class TestAssembleBriefing:
    def test_minimal_briefing(self) -> None:
        briefing = assemble_briefing(
            date="2025-01-30",
            volume=1,
            signals=[],
            tension_index={"composite": 0, "level": {"en": "Low", "zh": "低"}},
        )
        assert briefing["date"] == "2025-01-30"
        assert briefing["volume"] == 1
        assert briefing["signals"] == []
        assert "trade_data" in briefing
        assert "market_data" in briefing

    def test_with_signals(self) -> None:
        signals = [{"id": "test", "title": {"en": "Test", "zh": "测试"}}]
        briefing = assemble_briefing(
            date="2025-01-30",
            volume=2,
            signals=signals,
            tension_index={"composite": 5.0},
        )
        assert len(briefing["signals"]) == 1

    def test_default_structures(self) -> None:
        briefing = assemble_briefing(
            date="2025-01-30",
            volume=1,
            signals=[],
            tension_index={},
        )
        assert briefing["trade_data"]["summary_stats"] == []
        assert briefing["market_data"]["indices"] == []
        assert briefing["parliament"]["hansard"]["session_mentions"] == 0

    def test_optional_fields(self) -> None:
        briefing = assemble_briefing(
            date="2025-01-30",
            volume=1,
            signals=[],
            tension_index={},
            pathway_cards=[{"id": "test"}],
        )
        assert "pathway_cards" in briefing


class TestWriteProcessed:
    def test_writes_file(self, tmp_path: Path) -> None:
        briefing = {"date": "2025-01-30", "volume": 1}
        path = write_processed("2025-01-30", briefing, str(tmp_path))
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["date"] == "2025-01-30"

    def test_creates_latest(self, tmp_path: Path) -> None:
        briefing = {"date": "2025-01-30"}
        write_processed("2025-01-30", briefing, str(tmp_path))
        latest = tmp_path / "latest" / "briefing.json"
        assert latest.exists()


class TestWriteArchive:
    def test_writes_archive(self, tmp_path: Path) -> None:
        briefing = {"date": "2025-01-30"}
        path = write_archive("2025-01-30", briefing, str(tmp_path))
        assert path.exists()
        assert "daily" in str(path)


class TestValidateBriefing:
    def test_no_schema_dir(self) -> None:
        assert validate_briefing({}, schemas_dir="") is True

    def test_missing_schema_file(self, tmp_path: Path) -> None:
        assert validate_briefing({}, schemas_dir=str(tmp_path)) is True
