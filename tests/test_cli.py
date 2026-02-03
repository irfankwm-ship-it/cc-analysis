"""Tests for CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from analysis.cli import _score_sentence, _summarize_body, main


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def raw_data_dir(tmp_path: Path) -> Path:
    """Create a temporary raw data directory with sample signals."""
    raw_dir = tmp_path / "raw" / "2025-01-30"
    raw_dir.mkdir(parents=True)

    zh_body_1 = (
        "\u52A0\u62FF\u5927\u5168\u7403\u4E8B\u52A1\u90E8"
        "\u5C31\u5916\u4EA4\u7D27\u5F20\u5C40\u52BF"
        "\u53EC\u89C1\u4E2D\u56FD\u5927\u4F7F\u3002"
    )
    zh_body_2 = (
        "\u5546\u52A1\u90E8\u5BA3\u5E03\u5BF9\u52A0\u62FF"
        "\u5927\u6CB9\u83DC\u7C7D\u8FDB\u53E3\u52A0\u5F81"
        "\u65B0\u5173\u7A0E\u3002\u8D38\u6613\u6218"
        "\u5347\u7EA7\u7EE7\u7EED\u3002"
    )

    signals = {
        "metadata": {
            "fetch_timestamp": "2025-01-30T12:00:00Z",
            "source_name": "test",
            "version": "0.1.0",
            "date": "2025-01-30",
        },
        "data": {
            "articles": [
                {
                    "title": {
                        "en": "Canada Summons Chinese Ambassador",
                        "zh": "\u52A0\u62FF\u5927\u53EC\u89C1\u4E2D\u56FD\u5927\u4F7F",
                    },
                    "body": {
                        "en": (
                            "Global Affairs Canada summoned the"
                            " Chinese ambassador over diplomatic"
                            " tensions."
                        ),
                        "zh": zh_body_1,
                    },
                    "source": {
                        "en": "Global Affairs Canada",
                        "zh": "\u52A0\u62FF\u5927\u5168\u7403\u4E8B\u52A1\u90E8",
                    },
                    "date": "2025-01-30",
                    "implications": {
                        "canada_impact": {
                            "en": (
                                "Direct impact on bilateral"
                                " relations."
                            ),
                            "zh": (
                                "\u5BF9\u53CC\u8FB9\u5173\u7CFB"
                                "\u4EA7\u751F\u76F4\u63A5"
                                "\u5F71\u54CD\u3002"
                            ),
                        },
                    },
                },
                {
                    "title": {
                        "en": "China Imposes Canola Tariff",
                        "zh": (
                            "\u4E2D\u56FD\u5BF9\u6CB9\u83DC\u7C7D"
                            "\u52A0\u5F81\u5173\u7A0E"
                        ),
                    },
                    "body": {
                        "en": (
                            "MOFCOM announced new tariffs on"
                            " Canadian canola imports. Trade"
                            " war escalation continues."
                        ),
                        "zh": zh_body_2,
                    },
                    "source": {
                        "en": "MOFCOM",
                        "zh": "\u5546\u52A1\u90E8",
                    },
                    "date": "2025-01-30",
                    "implications": {
                        "canada_impact": {
                            "en": "Agricultural exports affected.",
                            "zh": (
                                "\u519C\u4EA7\u54C1\u51FA\u53E3"
                                "\u53D7\u5F71\u54CD\u3002"
                            ),
                        },
                    },
                },
            ]
        },
    }

    with open(raw_dir / "news.json", "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False)

    return raw_dir


class TestRunCommand:
    """Test the 'run' command."""

    def test_run_with_raw_data(
        self, runner: CliRunner, raw_data_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed"
        archive_dir = tmp_path / "archive"

        result = runner.invoke(main, [
            "run",
            "--env", "dev",
            "--date", "2025-01-30",
            "--raw-dir", str(raw_data_dir),
            "--output-dir", str(output_dir),
            "--archive-dir", str(archive_dir),
            "--schemas-dir", "",
        ])

        assert result.exit_code == 0
        assert "Analysis complete" in result.output

        # Check output was created
        briefing_path = output_dir / "2025-01-30" / "briefing.json"
        assert briefing_path.exists()

        with open(briefing_path) as f:
            briefing = json.load(f)

        assert briefing["date"] == "2025-01-30"
        assert len(briefing["signals"]) == 2
        assert "tension_index" in briefing
        assert briefing["tension_index"]["composite"] >= 0

    def test_run_empty_raw_dir(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "empty_raw"
        empty_dir.mkdir()
        output_dir = tmp_path / "processed"
        archive_dir = tmp_path / "archive"

        result = runner.invoke(main, [
            "run",
            "--date", "2025-01-30",
            "--raw-dir", str(empty_dir),
            "--output-dir", str(output_dir),
            "--archive-dir", str(archive_dir),
            "--schemas-dir", "",
        ])

        assert result.exit_code == 0

    def test_run_nonexistent_raw_dir(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed"
        archive_dir = tmp_path / "archive"

        result = runner.invoke(main, [
            "run",
            "--date", "2025-01-30",
            "--raw-dir", str(tmp_path / "nonexistent"),
            "--output-dir", str(output_dir),
            "--archive-dir", str(archive_dir),
            "--schemas-dir", "",
        ])

        assert result.exit_code == 0


class TestCompileVolumeCommand:
    """Test the 'compile-volume' command."""

    def test_compile_empty_archive(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(main, [
            "compile-volume",
            "--date", "2025-02-01",
            "--archive-dir", str(tmp_path),
        ])

        assert result.exit_code == 0
        assert "Volume" in result.output

    def test_compile_with_data(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Create archive data
        daily_dir = tmp_path / "daily" / "2025-01-15"
        daily_dir.mkdir(parents=True)

        briefing = {
            "date": "2025-01-15",
            "signals": [
                {"category": "trade", "severity": "high"},
            ],
            "tension_index": {"composite": 4.5},
        }
        with open(daily_dir / "briefing.json", "w") as f:
            json.dump(briefing, f)

        result = runner.invoke(main, [
            "compile-volume",
            "--date", "2025-02-01",
            "--archive-dir", str(tmp_path),
        ])

        assert result.exit_code == 0
        # Check volume file was created
        vol_file = tmp_path / "volumes" / "vol-001.json"
        assert vol_file.exists()


class TestHelpOutput:
    """Test CLI help messages."""

    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "China Compass analysis pipeline" in result.output

    def test_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--env" in result.output
        assert "--date" in result.output

    def test_compile_volume_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["compile-volume", "--help"])
        assert result.exit_code == 0
        assert "--archive-dir" in result.output

    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestScoreSentence:
    """Tests for _score_sentence title-relevance scoring."""

    def test_relevant_sentence_outscores_data_heavy_irrelevant(self) -> None:
        title = "India buying Venezuelan oil"
        relevant = (
            "India has increased purchases of Venezuelan crude"
            " by 32%, importing 200,000 barrels per day."
        )
        irrelevant = (
            "An armada of 12 warships is heading toward Iran,"
            " with 3,500 troops deployed across 5 countries."
        )
        score_relevant = _score_sentence(relevant, title, 0, 2)
        score_irrelevant = _score_sentence(irrelevant, title, 1, 2)
        assert score_relevant > score_irrelevant

    def test_zero_overlap_penalty_applied(self) -> None:
        title = "China imposes trade sanctions"
        with_overlap = "China announced new trade restrictions on imports."
        without_overlap = "The committee released a report on fiscal policy."
        score_with = _score_sentence(with_overlap, title, 0, 2)
        score_without = _score_sentence(without_overlap, title, 1, 2)
        assert score_with > score_without

    def test_no_penalty_when_title_has_no_matchable_words(self) -> None:
        title = "Oil"  # no words >= 4 chars
        sent = "The committee released a report on fiscal policy trends."
        score = _score_sentence(sent, title, 0, 1)
        # Without penalty, base score should not be dragged negative
        assert score >= -1.0  # only short-sentence penalty possible

    def test_summarize_body_prefers_relevant_content(self) -> None:
        title = "India buying Venezuelan oil"
        body = (
            "India has been buying Venezuelan crude oil at steep discounts "
            "under a complex arrangement involving intermediary traders, "
            "increasing total imports by 32% over the past quarter "
            "according to shipping data tracked by energy analysts. "
            "An armada of 12 warships, 45 aircraft, and 3,500 troops were "
            "deployed across 5 countries near Iran in a major military "
            "exercise that lasted several weeks. "
            "Officials confirmed that India will keep buying Venezuelan "
            "oil as part of its energy diversification strategy despite "
            "strong Western objections. "
            "Meanwhile a separate naval drill involved 8 destroyers and "
            "2 aircraft carriers conducting operations in the Persian Gulf "
            "with coalition forces from 7 nations. "
            "The Venezuelan government has welcomed India as a key buyer, "
            "with bilateral oil trade now exceeding four billion dollars "
            "annually. "
            "Pentagon officials said the military exercises were "
            "pre-planned and unrelated to any current geopolitical "
            "tensions in the region."
        )
        summary = _summarize_body(body, title)
        assert "India" in summary
        assert "armada" not in summary

    def test_summarize_body_fallback_when_no_overlap(self) -> None:
        title = "Xi PM"  # no words >= 4 chars
        body = (
            "The two leaders met at the summit to discuss bilateral trade "
            "and regional security. Officials announced a joint communique "
            "on economic cooperation between the nations."
        )
        summary = _summarize_body(body, title)
        assert len(summary) > 0
