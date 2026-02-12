"""Tests for signal_filtering module."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.signal_filtering import (
    _extract_signal_text,
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


class TestLowValuePatterns:
    def test_astronomy_negative_score(self) -> None:
        """Astronomy signals should get a low-value penalty."""
        signal = {
            "title": "Chinese scientists discover black hole devouring white dwarf star"
        }
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason

    def test_telescope_negative_score(self) -> None:
        signal = {"title": "New telescope at Beijing observatory spots galaxy cluster"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason

    def test_astrophysics_negative_score(self) -> None:
        signal = {"title": "Chinese astrophysics team discovers neutron star merger"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason


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


class TestExtractSignalText:
    def test_string_fields(self) -> None:
        signal = {"title": "Hello World", "body": "Some body text"}
        full, title = _extract_signal_text(signal)
        assert "hello world" in title
        assert "some body text" in full

    def test_bilingual_dict_fields(self) -> None:
        signal = {
            "title": {"en": "English title", "zh": "中文标题"},
            "body": {"en": "English body", "zh": "中文正文"},
        }
        full, title = _extract_signal_text(signal)
        assert "english title" in title
        assert "中文标题" in title
        assert "english body" in full
        assert "中文正文" in full

    def test_zh_only_string(self) -> None:
        signal = {"title": "加拿大与中国签署贸易协议", "body": "正文内容"}
        full, title = _extract_signal_text(signal)
        assert "加拿大" in title
        assert "正文内容" in full


class TestChineseLowValuePatterns:
    def test_weather_penalty(self) -> None:
        signal = {"title": "北京天气预报：明日暴雨预警"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason

    def test_entertainment_penalty(self) -> None:
        signal = {"title": "中国电影票房创新高 综艺节目收视率上升"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason

    def test_real_estate_penalty(self) -> None:
        signal = {"title": "上海楼市房价持续下跌 新楼盘预售不佳"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason

    def test_real_estate_policy_no_penalty(self) -> None:
        """Real estate with policy angle should NOT get penalty."""
        signal = {"title": "中国房地产政策调控新措施"}
        score, reason = compute_signal_value(signal)
        assert "low-value" not in reason

    def test_celebrity_penalty(self) -> None:
        signal = {"title": "中国明星八卦绯闻大爆料"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason

    def test_sports_penalty(self) -> None:
        signal = {"title": "中国运动员获得锦标赛冠军"}
        score, reason = compute_signal_value(signal)
        assert score < 0
        assert "low-value" in reason


class TestChineseBilateral:
    def test_bilateral_chinese_text(self) -> None:
        signal = {"title": "加拿大与中国签署新贸易协议"}
        assert is_bilateral(signal)

    def test_bilateral_chinese_body(self) -> None:
        signal = {"title": "新协议", "body": "加拿大总理访问北京讨论中国贸易问题"}
        assert is_bilateral(signal)

    def test_not_bilateral_chinese(self) -> None:
        signal = {"title": "中国发布新经济政策"}
        assert not is_bilateral(signal)

    def test_bilateral_bilingual_dict(self) -> None:
        signal = {
            "title": {"en": "Trade deal", "zh": "加拿大与中国贸易协议"},
            "body": {"en": "Details", "zh": "详情"},
        }
        assert is_bilateral(signal)

    def test_bilateral_title_boost_chinese(self) -> None:
        signal = {"title": "加拿大与中国达成重要协议"}
        score, reason = compute_signal_value(signal)
        assert score >= 3
        assert "bilateral in title" in reason

    def test_relevance_bilingual_dict_zh(self) -> None:
        """is_china_relevant checks ZH field of bilingual dicts."""
        signal = {
            "title": {"en": "Some unrelated title", "zh": "北京新政策"},
        }
        assert is_china_relevant(signal)


class TestFilterAndPrioritize:
    def test_basic_filtering(self) -> None:
        signals = [
            {"title": "Canada China talks", "date": "2025-01-30", "source": "Reuters"},
            {"title": "China policy update", "date": "2025-01-30", "source": "Xinhua"},
        ]
        result = filter_and_prioritize_signals(signals, "2025-01-30")
        assert len(result) <= 75
        assert len(result) >= 1
