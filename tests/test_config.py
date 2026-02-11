"""Tests for config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.config import (
    TemplateData,
    ThresholdsConfig,
    detect_env,
    load_config,
)


class TestDetectEnv:
    def test_explicit_env(self) -> None:
        assert detect_env("prod") == "prod"

    def test_default_env(self) -> None:
        assert detect_env(None) == "dev"

    def test_invalid_env(self) -> None:
        with pytest.raises(ValueError, match="Invalid environment"):
            detect_env("invalid")


class TestLoadConfig:
    def test_loads_dev_config(self) -> None:
        config = load_config("dev")
        assert config.env == "dev"
        assert config.paths.raw_dir == "../cc-data/raw"

    def test_loads_prod_config(self) -> None:
        config = load_config("prod")
        assert config.env == "prod"
        assert config.logging.level == "WARNING"

    def test_thresholds_loaded(self) -> None:
        config = load_config("dev")
        assert config.thresholds.dedup.title_exact_en == 0.85
        assert config.thresholds.filtering.min_signals == 10

    def test_templates_loaded(self) -> None:
        config = load_config("dev")
        assert len(config.templates.impact_templates) > 0
        assert "diplomatic" in config.templates.impact_templates

    def test_chinese_sources_loaded(self) -> None:
        config = load_config("dev")
        assert len(config.chinese_sources.source_names) > 0
        assert "xinhua" in config.chinese_sources.source_names

    def test_relevance_loaded(self) -> None:
        config = load_config("dev")
        assert len(config.relevance.china_relevance) > 0
        assert "china" in config.relevance.china_relevance

    def test_text_patterns_loaded(self) -> None:
        config = load_config("dev")
        assert len(config.text_patterns.filler_patterns) > 0

    def test_missing_config_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("dev", config_dir=Path("/nonexistent"))


class TestThresholdsConfig:
    def test_defaults(self) -> None:
        t = ThresholdsConfig()
        assert t.dedup.title_exact_en == 0.85
        assert t.filtering.max_signals == 75
        assert t.translation.body_truncate_chars == 500


class TestTemplateData:
    def test_empty_defaults(self) -> None:
        t = TemplateData()
        assert t.impact_templates == {}

    def test_with_data(self) -> None:
        t = TemplateData(
            impact_templates={"test": {"en": "a", "zh": "b"}},
        )
        assert "test" in t.impact_templates
