"""Tests for source-to-tier mapping."""

from __future__ import annotations

from typing import Any

from analysis.classifiers.source_mapper import map_signal_source_tier, map_source_tier


class TestMapSourceTier:
    """Test source name to tier mapping."""

    def test_official_sources(self) -> None:
        assert map_source_tier("Global Affairs Canada") == "official"
        assert map_source_tier("PMO") == "official"
        assert map_source_tier("State Council") == "official"
        assert map_source_tier("MOFCOM") == "official"
        assert map_source_tier("PBOC") == "official"
        assert map_source_tier("MFA") == "official"

    def test_wire_sources(self) -> None:
        assert map_source_tier("Reuters") == "wire"
        assert map_source_tier("AP") == "wire"
        assert map_source_tier("AFP") == "wire"
        assert map_source_tier("Bloomberg") == "wire"

    def test_specialist_sources(self) -> None:
        assert map_source_tier("CSIS") == "specialist"
        assert map_source_tier("Sinocism") == "specialist"
        assert map_source_tier("China Brief") == "specialist"
        assert map_source_tier("MERICS") == "specialist"

    def test_media_sources(self) -> None:
        assert map_source_tier("Globe and Mail") == "media"
        assert map_source_tier("CBC") == "media"
        assert map_source_tier("South China Morning Post") == "media"
        assert map_source_tier("Xinhua") == "media"

    def test_case_insensitive(self) -> None:
        assert map_source_tier("reuters") == "wire"
        assert map_source_tier("REUTERS") == "wire"
        assert map_source_tier("global affairs canada") == "official"

    def test_chinese_sources(self) -> None:
        assert map_source_tier("\u5916\u4EA4\u90E8") == "official"
        assert map_source_tier("\u5546\u52A1\u90E8") == "official"
        assert map_source_tier("\u8DEF\u900F\u793E") == "wire"
        assert map_source_tier("\u65B0\u534E\u793E") == "media"

    def test_unknown_source_defaults_to_media(self) -> None:
        assert map_source_tier("Unknown News Outlet") == "media"
        assert map_source_tier("Random Blog") == "media"

    def test_empty_source_defaults_to_media(self) -> None:
        assert map_source_tier("") == "media"

    def test_substring_matching(self) -> None:
        # "Reuters News Agency" should match "Reuters"
        assert map_source_tier("Reuters News Agency") == "wire"


class TestMapSignalSourceTier:
    """Test signal-level source tier extraction."""

    def test_bilingual_source(self) -> None:
        signal = {"source": {"en": "Reuters", "zh": "\u8DEF\u900F\u793E"}}
        assert map_signal_source_tier(signal) == "wire"

    def test_string_source(self) -> None:
        signal = {"source": "MOFCOM"}
        assert map_signal_source_tier(signal) == "official"

    def test_chinese_source_fallback(self) -> None:
        signal = {"source": {"en": "Some Unknown", "zh": "\u5916\u4EA4\u90E8"}}
        assert map_signal_source_tier(signal) == "official"

    def test_missing_source(self) -> None:
        signal: dict[str, Any] = {}
        assert map_signal_source_tier(signal) == "media"
