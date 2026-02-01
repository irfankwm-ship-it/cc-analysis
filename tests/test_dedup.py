"""Tests for signal deduplication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.dedup import (
    DedupStats,
    body_jaccard,
    deduplicate_signals,
    is_duplicate,
    load_recent_signals,
    normalize_text,
    normalize_url,
    title_similarity,
)


# ── normalize_text ──────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert normalize_text("U.S.-China trade: war!") == "uschina trade war"

    def test_collapses_whitespace(self):
        assert normalize_text("a  b   c") == "a b c"

    def test_empty_string(self):
        assert normalize_text("") == ""


# ── normalize_url ───────────────────────────────────────────────────────────

class TestNormalizeUrl:
    def test_strips_scheme_and_trailing_slash(self):
        result = normalize_url("https://example.com/article/123/")
        assert result == "example.com/article/123"

    def test_strips_query_params(self):
        result = normalize_url("https://scmp.com/news?utm_source=rss&ref=home")
        assert result == "scmp.com/news"

    def test_case_insensitive(self):
        assert normalize_url("HTTPS://Example.COM/Path") == "example.com/path"

    def test_empty_returns_empty(self):
        assert normalize_url("") == ""

    def test_http_and_https_match(self):
        a = normalize_url("http://example.com/article")
        b = normalize_url("https://example.com/article")
        assert a == b


# ── title_similarity ────────────────────────────────────────────────────────

class TestTitleSimilarity:
    def test_identical(self):
        assert title_similarity("china trade war", "china trade war") == 1.0

    def test_very_similar(self):
        a = normalize_text("China imposes new tariffs")
        b = normalize_text("China imposes new tariff on imports")
        assert title_similarity(a, b) >= 0.70

    def test_different(self):
        a = normalize_text("China trade war escalates")
        b = normalize_text("Nipah virus cases reported in India")
        assert title_similarity(a, b) < 0.30

    def test_empty_returns_zero(self):
        assert title_similarity("", "something") == 0.0
        assert title_similarity("something", "") == 0.0


# ── body_jaccard ────────────────────────────────────────────────────────────

class TestBodyJaccard:
    def test_identical(self):
        text = "China announced new semiconductor export restrictions"
        assert body_jaccard(text, text) == 1.0

    def test_same_story_rewritten(self):
        a = (
            "China announced semiconductor export controls targeting "
            "gallium and germanium materials used in chip manufacturing"
        )
        b = (
            "Beijing imposed export restrictions on semiconductor "
            "materials including gallium germanium for chip production"
        )
        score = body_jaccard(a, b)
        assert score >= 0.30  # substantial word overlap

    def test_different_stories(self):
        a = "China semiconductor export controls gallium germanium chips"
        b = "Canada parliament election foreign interference committee"
        assert body_jaccard(a, b) < 0.15

    def test_empty_returns_zero(self):
        assert body_jaccard("", "something about china") == 0.0
        assert body_jaccard("something", "") == 0.0


# ── is_duplicate ────────────────────────────────────────────────────────────

class TestIsDuplicate:
    def test_url_exact_match(self):
        a = {"title": "Article A", "source_url": "https://scmp.com/news/123"}
        b = {"title": "Completely different title", "url": "http://scmp.com/news/123"}
        is_dup, reason = is_duplicate(a, b)
        assert is_dup is True
        assert reason == "url"

    def test_title_exact_match(self):
        a = {"title": "China imposes new tariffs on Canadian canola"}
        b = {"title": "China imposes new tariffs on Canadian canola"}
        is_dup, reason = is_duplicate(a, b)
        assert is_dup is True
        assert reason == "title"

    def test_title_near_match(self):
        a = {"title": "China imposes new tariffs on Canadian canola exports"}
        b = {"title": "China imposes new tariff on Canadian canola"}
        is_dup, reason = is_duplicate(a, b)
        assert is_dup is True
        assert reason == "title"

    def test_title_body_match(self):
        shared_body = (
            "The Chinese government announced new semiconductor export "
            "controls affecting gallium and germanium shipments to "
            "multiple countries including Canada and the United States"
        )
        a = {
            "title": "China restricts semiconductor exports to multiple countries",
            "body_text": shared_body,
        }
        b = {
            "title": "China restricts semiconductor material exports globally",
            "body": shared_body,
        }
        is_dup, reason = is_duplicate(a, b)
        assert is_dup is True
        assert reason == "title+body"

    def test_new_development_kept(self):
        a = {
            "title": "China tariffs on canola raised further",
            "body_text": (
                "Beijing increased the tariff rate from 100% to 150% "
                "in an escalation that surprised markets. The new rate "
                "takes effect immediately."
            ),
        }
        b = {
            "title": "China tariffs on canola imports begin",
            "body": (
                "China imposed a 100% tariff on Canadian canola imports "
                "citing quality concerns. The Canadian government called "
                "the move unjustified."
            ),
        }
        is_dup, reason = is_duplicate(a, b)
        assert is_dup is False
        assert reason == ""

    def test_completely_different(self):
        a = {"title": "Taiwan strait military exercises intensify"}
        b = {"title": "Canadian parliament debates foreign interference bill"}
        is_dup, reason = is_duplicate(a, b)
        assert is_dup is False
        assert reason == ""

    def test_bilingual_vs_raw(self):
        """Archived bilingual signal vs raw fetcher signal."""
        archived = {
            "title": {"en": "China imposes new tariffs on canola", "zh": "中国对油菜籽加征关税"},
            "body": {"en": "The tariff took effect Monday.", "zh": "关税周一生效。"},
        }
        raw = {
            "title": "China imposes new tariffs on canola",
            "body_snippet": "The tariff took effect Monday.",
        }
        is_dup, reason = is_duplicate(raw, archived)
        assert is_dup is True
        assert reason == "title"


# ── load_recent_signals ─────────────────────────────────────────────────────

class TestLoadRecentSignals:
    def _write_briefing(self, path: Path, signals: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"signals": signals}), encoding="utf-8")

    def test_loads_from_processed(self, tmp_path):
        proc = tmp_path / "processed"
        self._write_briefing(
            proc / "2026-01-31" / "briefing.json",
            [{"title": "Signal A"}],
        )
        result = load_recent_signals(
            str(proc), str(tmp_path / "archive"), "2026-02-01", lookback_days=1,
        )
        assert len(result) == 1
        assert result[0]["title"] == "Signal A"

    def test_falls_back_to_archive(self, tmp_path):
        archive = tmp_path / "archive"
        self._write_briefing(
            archive / "daily" / "2026-01-31" / "briefing.json",
            [{"title": "Archived Signal"}],
        )
        result = load_recent_signals(
            str(tmp_path / "processed"), str(archive), "2026-02-01", lookback_days=1,
        )
        assert len(result) == 1
        assert result[0]["title"] == "Archived Signal"

    def test_loads_multiple_days(self, tmp_path):
        proc = tmp_path / "processed"
        self._write_briefing(
            proc / "2026-01-31" / "briefing.json",
            [{"title": "Day 1 signal"}],
        )
        self._write_briefing(
            proc / "2026-01-30" / "briefing.json",
            [{"title": "Day 2 signal"}],
        )
        result = load_recent_signals(
            str(proc), str(tmp_path / "archive"), "2026-02-01", lookback_days=3,
        )
        assert len(result) == 2

    def test_handles_missing_dates(self, tmp_path):
        """Gaps in archive should not cause errors."""
        proc = tmp_path / "processed"
        self._write_briefing(
            proc / "2026-01-30" / "briefing.json",
            [{"title": "Day 2 signal"}],
        )
        # Jan 31 is missing — should still return Day 2 signal
        result = load_recent_signals(
            str(proc), str(tmp_path / "archive"), "2026-02-01", lookback_days=3,
        )
        assert len(result) == 1

    def test_handles_corrupted_json(self, tmp_path):
        proc = tmp_path / "processed" / "2026-01-31"
        proc.mkdir(parents=True)
        (proc / "briefing.json").write_text("{invalid json", encoding="utf-8")
        result = load_recent_signals(
            str(tmp_path / "processed"), str(tmp_path / "archive"), "2026-02-01",
        )
        assert result == []

    def test_invalid_date_returns_empty(self, tmp_path):
        result = load_recent_signals(
            str(tmp_path), str(tmp_path), "not-a-date",
        )
        assert result == []


# ── deduplicate_signals ─────────────────────────────────────────────────────

class TestDeduplicateSignals:
    def test_within_day_url_dedup(self):
        signals = [
            {"title": "Article A", "source_url": "https://scmp.com/123"},
            {"title": "Different title", "source_url": "https://scmp.com/123"},
        ]
        result, stats = deduplicate_signals(signals)
        assert len(result) == 1
        assert result[0]["title"] == "Article A"
        assert stats.dropped_url == 1

    def test_within_day_title_dedup(self):
        signals = [
            {"title": "China imposes tariffs on canola"},
            {"title": "China imposes tariffs on canola imports"},
        ]
        result, stats = deduplicate_signals(signals)
        assert len(result) == 1
        assert stats.dropped_title == 1

    def test_cross_day_dedup(self):
        current = [
            {"title": "China announces semiconductor restrictions"},
        ]
        previous = [
            {"title": {"en": "China announces semiconductor restrictions", "zh": ""}},
        ]
        result, stats = deduplicate_signals(current, previous)
        assert len(result) == 0
        assert stats.dropped_title == 1

    def test_keeps_new_developments(self):
        current = [
            {
                "title": "Taiwan strait tensions escalate with new exercises",
                "body_text": (
                    "PLA forces conducted live-fire drills near Taiwan for "
                    "the third consecutive day, deploying additional naval "
                    "vessels and aircraft in an unprecedented show of force."
                ),
            },
        ]
        previous = [
            {
                "title": {"en": "Taiwan strait military exercises begin", "zh": ""},
                "body": {
                    "en": (
                        "China launched military exercises around Taiwan in "
                        "response to a senior US official's visit to Taipei."
                    ),
                    "zh": "",
                },
            },
        ]
        result, stats = deduplicate_signals(current, previous)
        assert len(result) == 1
        assert stats.total_dropped == 0

    def test_empty_input(self):
        result, stats = deduplicate_signals([])
        assert result == []
        assert stats.total_before == 0
        assert stats.total_after == 0

    def test_stats_correct(self):
        signals = [
            {"title": "Unique signal about Taiwan"},
            {"title": "Unique signal about Taiwan tensions"},  # similar title
            {"title": "Completely different signal about trade"},
        ]
        result, stats = deduplicate_signals(signals)
        assert stats.total_before == 3
        assert stats.total_after == len(result)
        assert stats.total_dropped == stats.total_before - stats.total_after

    def test_preserves_order(self):
        signals = [
            {"title": "Taiwan strait military exercises intensify"},
            {"title": "Canada parliament debates foreign interference"},
            {"title": "PBOC cuts interest rates to boost economy"},
        ]
        result, _ = deduplicate_signals(signals)
        assert len(result) == 3
        assert result[0]["title"] == "Taiwan strait military exercises intensify"
        assert result[1]["title"] == "Canada parliament debates foreign interference"
        assert result[2]["title"] == "PBOC cuts interest rates to boost economy"

    def test_no_previous_signals(self):
        signals = [
            {"title": "Taiwan strait military exercises"},
            {"title": "Canada parliament foreign interference bill"},
        ]
        result, stats = deduplicate_signals(signals, None)
        assert len(result) == 2
        assert stats.total_dropped == 0
