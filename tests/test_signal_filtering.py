"""Tests for signal_filtering module."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.signal_filtering import (
    compute_signal_value,
    filter_and_prioritize_signals,
    filter_low_value_signals,
    is_bilateral,
    is_china_relevant,
    load_raw_signals,
    parse_signal_date,
)


class TestLoadRawSignals:
    def test_load_from_directory(self, tmp_path: Path) -> None:
        data = {"data": {"articles": [{"title": "Test signal"}]}}
        with open(tmp_path / "news.json", "w") as f:
            json.dump(data, f)
        signals = load_raw_signals(str(tmp_path))
        assert len(signals) == 1
        assert signals[0]["title"] == "Test signal"

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        signals = load_raw_signals(str(tmp_path / "nonexistent"))
        assert signals == []

    def test_handles_list_payload(self, tmp_path: Path) -> None:
        data = [{"title": "Signal 1"}, {"title": "Signal 2"}]
        with open(tmp_path / "signals.json", "w") as f:
            json.dump(data, f)
        signals = load_raw_signals(str(tmp_path))
        assert len(signals) == 2

    def test_handles_malformed_json(self, tmp_path: Path) -> None:
        with open(tmp_path / "bad.json", "w") as f:
            f.write("not json")
        signals = load_raw_signals(str(tmp_path))
        assert signals == []


class TestParseSignalDate:
    def test_iso_date(self) -> None:
        dt = parse_signal_date({"date": "2025-01-30"})
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 30

    def test_rss_date(self) -> None:
        dt = parse_signal_date({"date": "Thu, 30 Jan 2025 12:00:00 +0000"})
        assert dt is not None
        assert dt.day == 30

    def test_bilingual_date(self) -> None:
        dt = parse_signal_date({"date": {"en": "2025-01-30", "zh": "2025-01-30"}})
        assert dt is not None

    def test_empty_date(self) -> None:
        assert parse_signal_date({"date": ""}) is None
        assert parse_signal_date({}) is None


class TestIsChinaRelevant:
    def test_china_in_title(self) -> None:
        assert is_china_relevant({"title": "China imposes new tariffs"})

    def test_chinese_keyword(self) -> None:
        assert is_china_relevant({"title": "习近平会见外宾"})

    def test_irrelevant_signal(self) -> None:
        assert not is_china_relevant({"title": "Local weather report"})

    def test_bilingual_title(self) -> None:
        assert is_china_relevant({"title": {"en": "Beijing trade talks", "zh": ""}})


class TestComputeSignalValue:
    def test_bilateral_boosts_score(self) -> None:
        signal = {"title": "Canada and China reach trade deal"}
        score, _ = compute_signal_value(signal)
        assert score >= 3

    def test_low_value_penalty(self) -> None:
        signal = {"title": "Celebrity gossip about China star"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason

    def test_high_value_keywords(self) -> None:
        signal = {"title": "Xi Jinping announces sanctions policy"}
        score, _ = compute_signal_value(signal)
        assert score >= 1

    def test_canadian_source_boost(self) -> None:
        signal = {"title": "China news", "source": "CBC"}
        score, reason = compute_signal_value(signal)
        assert "Canadian source" in reason


class TestFilterLowValueSignals:
    def test_filters_low_value(self) -> None:
        signals = [
            {"title": "Canada China trade deal sanctions tariff"},
            {"title": "Celebrity gossip dating romance"},
        ]
        result = filter_low_value_signals(signals, min_score=0)
        assert len(result) >= 1


class TestIsBilateral:
    def test_bilateral(self) -> None:
        assert is_bilateral({"title": "Canada and China sign trade agreement"})

    def test_not_bilateral(self) -> None:
        assert not is_bilateral({"title": "China announces new policy"})

    def test_custom_keywords(self) -> None:
        assert is_bilateral(
            {"title": "Ottawa Beijing talks"},
            canada_keywords=["ottawa"],
            china_keywords=["beijing"],
        )


class TestFilterAndPrioritize:
    def test_basic_filtering(self) -> None:
        signals = [
            {"title": "Canada China talks", "date": "2025-01-30", "source": "Reuters"},
            {"title": "China policy update", "date": "2025-01-30", "source": "Xinhua"},
        ]
        result = filter_and_prioritize_signals(signals, "2025-01-30")
        assert len(result) <= 75
        assert len(result) >= 1
