"""Tests for signal deduplication."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.dedup import (
    DEFAULT_LOOKBACK_DAYS,
    TITLE_EXACT_THRESHOLD_EN,
    TITLE_EXACT_THRESHOLD_ZH,
    _contains_chinese,
    _detect_language,
    _extract_category,
    _extract_entities,
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


# ── Chinese language support ───────────────────────────────────────────────

class TestChineseSupport:
    def test_contains_chinese_true(self):
        assert _contains_chinese("中国外交部发言人") is True
        assert _contains_chinese("China 中国") is True

    def test_contains_chinese_false(self):
        assert _contains_chinese("China trade war") is False
        assert _contains_chinese("") is False

    def test_detect_language_chinese(self):
        assert _detect_language("中国对加拿大实施制裁") == "zh"
        assert _detect_language("China announces 中国宣布 new policy") == "zh"

    def test_detect_language_english(self):
        assert _detect_language("China announces new trade policy") == "en"
        assert _detect_language("") == "en"

    def test_chinese_title_lower_threshold(self):
        """Chinese titles use 0.70 threshold vs 0.85 for English."""
        assert TITLE_EXACT_THRESHOLD_ZH < TITLE_EXACT_THRESHOLD_EN
        assert TITLE_EXACT_THRESHOLD_ZH == 0.70
        assert TITLE_EXACT_THRESHOLD_EN == 0.85  # Relaxed for more signal throughput

    def test_chinese_similar_titles_dedup(self):
        """Chinese titles with ~75% similarity should dedup."""
        a = {"title": "中国外交部就加拿大涉台言论表示强烈不满"}
        b = {"title": "中国外交部就加拿大涉台言论表态"}
        is_dup, reason = is_duplicate(a, b)
        # These should match under the lower Chinese threshold
        assert is_dup is True
        assert reason == "title"

    def test_chinese_different_stories_kept(self):
        """Genuinely different Chinese stories should be kept."""
        a = {"title": "中国外交部就加拿大涉台言论表态"}
        b = {"title": "商务部公布对加拿大油菜籽反倾销调查结果"}
        is_dup, _ = is_duplicate(a, b)
        assert is_dup is False

    def test_chinese_stop_words_excluded(self):
        """Chinese stop words should not inflate Jaccard similarity."""
        # Same substantive content with different function words
        a = "中国政府宣布对加拿大实施新的贸易制裁措施"
        b = "中国政府宣布将对加拿大实施新贸易制裁"
        score = body_jaccard(a, b)
        # Should have high overlap on content words
        assert score >= 0.40

    def test_mixed_language_dedup(self):
        """English headline vs Chinese translation should not false-positive."""
        a = {"title": "China announces new tariffs on Canadian canola"}
        b = {"title": "中国宣布对加拿大油菜籽加征关税"}
        is_dup, _ = is_duplicate(a, b)
        # Different scripts, shouldn't match on title alone
        # (would need URL or body match)
        assert is_dup is False


# ── Entity-based deduplication ─────────────────────────────────────────────

class TestEntityDedup:
    def test_extract_entities_raw_format(self):
        signal = {"entities": ["xi_jinping", "mfa", "canada"]}
        entities = _extract_entities(signal)
        assert entities == {"xi_jinping", "mfa", "canada"}

    def test_extract_entities_processed_format(self):
        signal = {"entity_ids": ["huawei", "mofcom"]}
        entities = _extract_entities(signal)
        assert entities == {"huawei", "mofcom"}

    def test_extract_entities_dict_format(self):
        signal = {"entities": [{"id": "taiwan"}, {"id": "pla"}]}
        entities = _extract_entities(signal)
        assert entities == {"taiwan", "pla"}

    def test_extract_entities_empty(self):
        assert _extract_entities({}) == set()
        assert _extract_entities({"entities": []}) == set()

    def test_extract_category(self):
        assert _extract_category({"category": "diplomatic"}) == "diplomatic"
        assert _extract_category({"category": {"en": "Trade", "zh": "贸易"}}) == "trade"
        assert _extract_category({}) == ""

    def test_entity_body_dedup(self):
        """Same entities + same category + similar body = duplicate."""
        shared_body = (
            "The Chinese Ministry of Foreign Affairs spokesperson criticized "
            "Canada's stance on Taiwan, calling it interference in internal affairs. "
            "Wang Yi met with Canadian officials to discuss bilateral relations."
        )
        a = {
            "title": "MFA slams Canada over Taiwan stance",
            "body_text": shared_body,
            "entities": ["mfa", "wang_yi", "taiwan"],
            "category": "diplomatic",
        }
        b = {
            "title": "China foreign ministry criticizes Ottawa",
            "body": shared_body[:200],  # Similar but not identical body
            "entity_ids": ["mfa", "wang_yi", "taiwan"],
            "category": "diplomatic",
        }
        is_dup, reason = is_duplicate(a, b)
        assert is_dup is True
        assert reason == "entity+body"

    def test_entity_dedup_requires_category_match(self):
        """Same entities but different category should not dedup."""
        a = {
            "title": "Huawei announces new 5G chip",
            "body_text": "Huawei unveiled its latest semiconductor technology.",
            "entities": ["huawei"],
            "category": "technology",
        }
        b = {
            "title": "Huawei faces new trade restrictions",
            "body": "Huawei was added to the export control list.",
            "entity_ids": ["huawei"],
            "category": "trade",
        }
        is_dup, reason = is_duplicate(a, b)
        # Different categories, different stories
        assert is_dup is False

    def test_entity_dedup_requires_body_overlap(self):
        """Same entities + category but different body = not duplicate."""
        a = {
            "title": "MOFCOM announces export controls",
            "body_text": (
                "The Ministry of Commerce published new regulations on "
                "rare earth exports effective immediately."
            ),
            "entities": ["mofcom", "rare_earths"],
            "category": "trade",
        }
        b = {
            "title": "MOFCOM holds press conference",
            "body": (
                "The Ministry of Commerce discussed ongoing trade "
                "negotiations with European partners on steel tariffs."
            ),
            "entity_ids": ["mofcom"],
            "category": "trade",
        }
        is_dup, _ = is_duplicate(a, b)
        # Same MOFCOM entity but completely different topic
        assert is_dup is False

    def test_entity_dedup_stats(self):
        """Entity+body dedup should be tracked in stats."""
        # Use very different titles so title+body doesn't trigger first
        signals = [
            {
                "title": "Xi warns about economic headwinds",
                "body_text": "Spokesperson condemned interference in Taiwan affairs.",
                "entities": ["mfa", "taiwan"],
                "category": "diplomatic",
            },
            {
                "title": "Press briefing on regional tensions",
                "body_text": "Spokesperson condemned interference in Taiwan affairs.",
                "entities": ["mfa", "taiwan"],
                "category": "diplomatic",
            },
        ]
        result, stats = deduplicate_signals(signals)
        assert len(result) == 1
        assert stats.dropped_entity_body == 1


# ── Lookback configuration ─────────────────────────────────────────────────

class TestCustomThresholds:
    """Test that custom thresholds are accepted and used."""

    def test_strict_threshold_keeps_both(self):
        """Raising title threshold should keep signals that would otherwise dedup."""
        a = {"title": "China imposes tariffs on canola"}
        b = {"title": "China imposes tariffs on canola imports"}
        # With default threshold (0.85), these dedup
        is_dup_default, _ = is_duplicate(a, b)
        assert is_dup_default is True
        # With very strict threshold (0.99), they should be kept as distinct
        is_dup_strict, _ = is_duplicate(a, b, title_exact_en=0.99)
        assert is_dup_strict is False

    def test_loose_body_threshold(self):
        """Lowering body Jaccard threshold catches more duplicates."""
        a = {
            "title": "China trade policy changes in Asia Pacific region",
            "body_text": (
                "China announced policy changes affecting trade with "
                "multiple partners in the Asia Pacific region."
            ),
        }
        b = {
            "title": "China trade policy updated for regional partners",
            "body": (
                "China updated its trade policy for partners in the "
                "Asia Pacific zone with new regulations."
            ),
        }
        # With strict body threshold, these might be distinct
        is_dup_strict, _ = is_duplicate(a, b, body_jaccard_threshold=0.90)
        assert is_dup_strict is False
        # With loose body threshold, they should dedup
        is_dup_loose, _ = is_duplicate(a, b, body_jaccard_threshold=0.15)
        assert is_dup_loose is True

    def test_deduplicate_signals_accepts_thresholds(self):
        """deduplicate_signals forwards thresholds to is_duplicate."""
        signals = [
            {"title": "China imposes tariffs on canola"},
            {"title": "China imposes tariffs on canola imports"},
        ]
        # Default: dedup
        result_default, stats_default = deduplicate_signals(signals)
        assert len(result_default) == 1
        # Very strict: keep both
        result_strict, stats_strict = deduplicate_signals(
            signals, title_exact_en=0.99,
        )
        assert len(result_strict) == 2
        assert stats_strict.total_dropped == 0


class TestLookbackConfig:
    def test_default_lookback_is_seven_days(self):
        """Default lookback extended to 7 days to prevent story repetition."""
        assert DEFAULT_LOOKBACK_DAYS == 7
