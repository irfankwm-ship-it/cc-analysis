"""Tests for timeline_compiler module."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.timeline_compiler import (
    _create_empty_timeline,
    _deduplicate_against_existing,
    _deduplicate_events,
    _ensure_bilingual,
    _extract_tags,
    _get_en,
    _has_valid_translation,
    _is_chinese_text,
    _is_likely_french,
    _signal_to_event,
    _title_similarity,
    compile_canada_china_timeline,
    mark_signal_as_milestone,
    write_timeline,
)


class TestIsChineseText:
    def test_chinese(self) -> None:
        assert _is_chinese_text("这是中文")

    def test_english(self) -> None:
        assert not _is_chinese_text("This is English")

    def test_empty(self) -> None:
        assert not _is_chinese_text("")


class TestIsLikelyFrench:
    def test_french(self) -> None:
        assert _is_likely_french("Le gouvernement du Canada dans les affaires")

    def test_english(self) -> None:
        assert not _is_likely_french("The government of Canada in affairs")

    def test_empty(self) -> None:
        assert not _is_likely_french("")


class TestHasValidTranslation:
    def test_valid(self) -> None:
        assert _has_valid_translation({"en": "English text", "zh": "中文文本"})

    def test_missing_chinese(self) -> None:
        assert not _has_valid_translation({"en": "English", "zh": ""})

    def test_no_chinese_chars(self) -> None:
        assert not _has_valid_translation({"en": "English", "zh": "Still English"})

    def test_french_source(self) -> None:
        assert not _has_valid_translation({"en": "Le gouvernement du Canada dans", "zh": "中文"})


class TestTitleSimilarity:
    def test_identical(self) -> None:
        assert _title_similarity("Hello World", "Hello World") == 1.0

    def test_similar(self) -> None:
        assert _title_similarity("China Trade War", "China Trade Wars") > 0.8

    def test_different(self) -> None:
        assert _title_similarity("Hello", "Goodbye World") < 0.5

    def test_empty(self) -> None:
        assert _title_similarity("", "Hello") == 0.0


class TestDeduplicateEvents:
    def test_removes_duplicates(self) -> None:
        events = [
            {"date": "2025-01-30", "title": {"en": "China Trade War Escalates"}},
            {"date": "2025-01-30", "title": {"en": "China Trade War Escalating"}},
            {"date": "2025-01-31", "title": {"en": "Different Event"}},
        ]
        result = _deduplicate_events(events)
        assert len(result) == 2

    def test_keeps_different_dates(self) -> None:
        events = [
            {"date": "2025-01-30", "title": {"en": "Same Title"}},
            {"date": "2025-01-31", "title": {"en": "Same Title"}},
        ]
        result = _deduplicate_events(events)
        assert len(result) == 2

    def test_empty(self) -> None:
        assert _deduplicate_events([]) == []


class TestDeduplicateAgainstExisting:
    def test_removes_existing(self) -> None:
        new = [{"date": "2025-01-30", "title": {"en": "Existing Event"}}]
        existing = [{"date": "2025-01-30", "title": {"en": "Existing Event"}}]
        result = _deduplicate_against_existing(new, existing)
        assert len(result) == 0

    def test_keeps_new(self) -> None:
        new = [{"date": "2025-01-30", "title": {"en": "New Event"}}]
        existing = [{"date": "2025-01-30", "title": {"en": "Different Event"}}]
        result = _deduplicate_against_existing(new, existing)
        assert len(result) == 1


class TestSignalToEvent:
    def test_basic_conversion(self) -> None:
        signal = {
            "id": "test-signal",
            "title": {"en": "Test", "zh": "测试"},
            "body": {"en": "Body", "zh": "正文"},
            "category": "trade",
            "severity": "high",
            "is_milestone": True,
            "date": "2025-01-30",
        }
        event = _signal_to_event(signal, "2025-01-30")
        assert event["id"] == "test-signal"
        assert event["category"] == "trade"
        assert event["is_milestone"] is True


class TestExtractTags:
    def test_extracts_tags(self) -> None:
        signal = {
            "category": "trade",
            "entity_ids": ["canola", "huawei"],
            "severity": "critical",
        }
        tags = _extract_tags(signal)
        assert "trade" in tags
        assert "canola" in tags
        assert "severity-critical" in tags


class TestEnsureBilingual:
    def test_dict_input(self) -> None:
        assert _ensure_bilingual({"en": "Hello", "zh": "你好"})["en"] == "Hello"

    def test_string_input(self) -> None:
        assert _ensure_bilingual("Hello") == {"en": "Hello", "zh": "Hello"}


class TestGetEn:
    def test_dict(self) -> None:
        assert _get_en({"en": "Hello", "zh": "你好"}) == "Hello"

    def test_string(self) -> None:
        assert _get_en("Hello") == "Hello"


class TestCreateEmptyTimeline:
    def test_structure(self) -> None:
        t = _create_empty_timeline("test")
        assert t["id"] == "test"
        assert t["events"] == []
        assert t["metadata"]["total_events"] == 0


class TestCompileTimeline:
    def test_compile_from_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / "daily" / "2025-01-30"
        archive.mkdir(parents=True)
        timelines = tmp_path / "timelines"
        timelines.mkdir()

        briefing = {
            "signals": [
                {
                    "id": "test-signal",
                    "title": {"en": "Critical Event", "zh": "重大事件"},
                    "body": {"en": "Body", "zh": "正文"},
                    "category": "trade",
                    "severity": "critical",
                    "date": "2025-01-30",
                }
            ],
            "tension_index": {"composite": 5.0, "level": {"en": "Elevated"}},
        }
        with open(archive / "briefing.json", "w") as f:
            json.dump(briefing, f)

        timeline = compile_canada_china_timeline(
            str(tmp_path), str(timelines)
        )
        assert timeline["metadata"]["total_events"] >= 1

    def test_empty_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / "daily"
        archive.mkdir(parents=True)
        timelines = tmp_path / "timelines"
        timelines.mkdir()

        timeline = compile_canada_china_timeline(
            str(tmp_path), str(timelines)
        )
        assert timeline["metadata"]["total_events"] == 0


class TestWriteTimeline:
    def test_writes_file(self, tmp_path: Path) -> None:
        timeline = _create_empty_timeline("test")
        path = write_timeline(timeline, str(tmp_path))
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["id"] == "test"


class TestMarkMilestone:
    def test_marks_signal(self, tmp_path: Path) -> None:
        daily = tmp_path / "daily" / "2025-01-30"
        daily.mkdir(parents=True)
        briefing = {
            "signals": [{"id": "test-signal", "is_milestone": False}]
        }
        with open(daily / "briefing.json", "w") as f:
            json.dump(briefing, f)

        success = mark_signal_as_milestone("test-signal", "crisis", str(tmp_path))
        assert success is True

        with open(daily / "briefing.json") as f:
            data = json.load(f)
        assert data["signals"][0]["is_milestone"] is True
        assert data["signals"][0]["timeline_category"] == "crisis"

    def test_signal_not_found(self, tmp_path: Path) -> None:
        daily = tmp_path / "daily" / "2025-01-30"
        daily.mkdir(parents=True)
        briefing = {"signals": [{"id": "other"}]}
        with open(daily / "briefing.json", "w") as f:
            json.dump(briefing, f)

        assert mark_signal_as_milestone("nonexistent", archive_dir=str(tmp_path)) is False

    def test_no_archive_dir(self) -> None:
        assert mark_signal_as_milestone("test") is False
