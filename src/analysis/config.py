"""Environment-aware configuration loader.

Loads YAML config from config/analysis.{env}.yaml and keyword
dictionaries from config/keyword_dicts/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_ENVS = ("dev", "staging", "prod")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths for data I/O."""

    raw_dir: str
    processed_dir: str
    archive_dir: str
    schemas_dir: str


@dataclass(frozen=True)
class TensionConfig:
    """Configuration for tension index computation."""

    window_days: int = 30
    cap_denominator: int = 20


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass(frozen=True)
class ValidationConfig:
    """Schema validation settings."""

    strict: bool = True
    schema_file: str = "briefing.schema.json"


@dataclass(frozen=True)
class KeywordDicts:
    """Loaded keyword dictionaries."""

    categories: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    severity_modifiers: dict[str, Any] = field(default_factory=dict)
    entity_aliases: dict[str, dict[str, list[str]]] = field(default_factory=dict)


# --- Threshold dataclasses ---


@dataclass(frozen=True)
class DedupThresholds:
    """Deduplication thresholds."""

    title_exact_en: float = 0.85
    title_exact_zh: float = 0.70
    title_fuzzy_low: float = 0.35
    body_jaccard: float = 0.50
    entity_body_jaccard: float = 0.40
    lookback_days: int = 7


@dataclass(frozen=True)
class TextProcessingThresholds:
    """Text processing thresholds."""

    summary_max_chars: int = 500
    min_sentence_len: int = 15


@dataclass(frozen=True)
class FilteringThresholds:
    """Signal filtering thresholds."""

    recency_windows_hours: tuple[int, ...] = (72, 168)
    min_signals: int = 10
    max_signals: int = 75
    max_per_source: int = 3


@dataclass(frozen=True)
class TranslationThresholds:
    """Translation quality thresholds."""

    body_truncate_chars: int = 500
    english_fragment_threshold: float = 0.15


@dataclass(frozen=True)
class ThresholdsConfig:
    """All configurable thresholds."""

    dedup: DedupThresholds = field(default_factory=DedupThresholds)
    text_processing: TextProcessingThresholds = field(default_factory=TextProcessingThresholds)
    filtering: FilteringThresholds = field(default_factory=FilteringThresholds)
    translation: TranslationThresholds = field(default_factory=TranslationThresholds)


# --- Template / data config dataclasses ---


@dataclass(frozen=True)
class TemplateData:
    """Loaded implication and perspective templates."""

    impact_templates: dict[str, dict[str, str]] = field(default_factory=dict)
    watch_templates: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    canada_perspective: dict[str, dict[str, str]] = field(default_factory=dict)
    china_perspective: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ChineseSourceData:
    """Chinese source detection data."""

    source_names: frozenset[str] = field(default_factory=frozenset)
    domains: frozenset[str] = field(default_factory=frozenset)
    name_translations: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RelevanceData:
    """Relevance and value-scoring keyword lists."""

    china_relevance: list[str] = field(default_factory=list)
    low_value_patterns: list[str] = field(default_factory=list)
    high_value_keywords: list[str] = field(default_factory=list)
    canada_keywords: list[str] = field(default_factory=list)
    china_keywords: list[str] = field(default_factory=list)
    canadian_sources: frozenset[str] = field(default_factory=frozenset)
    regulatory_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TextPatternData:
    """Text processing pattern lists."""

    filler_patterns: list[str] = field(default_factory=list)
    key_point_patterns: list[str] = field(default_factory=list)
    boilerplate_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    env: str
    paths: PathsConfig
    tension: TensionConfig
    logging: LoggingConfig
    validation: ValidationConfig
    keywords: KeywordDicts
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    templates: TemplateData = field(default_factory=TemplateData)
    chinese_sources: ChineseSourceData = field(default_factory=ChineseSourceData)
    relevance: RelevanceData = field(default_factory=RelevanceData)
    text_patterns: TextPatternData = field(default_factory=TextPatternData)


def detect_env(cli_env: str | None = None) -> str:
    """Detect the runtime environment.

    Priority:
      1. Explicit CLI flag
      2. CC_ENV environment variable
      3. Default to 'dev'
    """
    env = cli_env or os.environ.get("CC_ENV", "dev")
    if env not in VALID_ENVS:
        raise ValueError(f"Invalid environment '{env}'. Must be one of {VALID_ENVS}")
    return env


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_yaml_optional(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_keyword_dicts(config_dir: Path) -> KeywordDicts:
    """Load all keyword dictionary YAML files."""
    kw_dir = config_dir / "keyword_dicts"

    categories_path = kw_dir / "categories.yaml"
    severity_path = kw_dir / "severity_modifiers.yaml"
    entity_path = kw_dir / "entity_aliases.yaml"

    categories: dict[str, dict[str, list[str]]] = {}
    if categories_path.exists():
        categories = _load_yaml(categories_path)

    severity_modifiers: dict[str, Any] = {}
    if severity_path.exists():
        severity_modifiers = _load_yaml(severity_path)

    entity_aliases: dict[str, dict[str, list[str]]] = {}
    if entity_path.exists():
        entity_aliases = _load_yaml(entity_path)

    return KeywordDicts(
        categories=categories,
        severity_modifiers=severity_modifiers,
        entity_aliases=entity_aliases,
    )


def _load_thresholds(raw: dict[str, Any]) -> ThresholdsConfig:
    """Parse thresholds section from env config."""
    t = raw.get("thresholds", {})
    if not t:
        return ThresholdsConfig()

    d = t.get("dedup", {})
    tp = t.get("text_processing", {})
    f = t.get("filtering", {})
    tr = t.get("translation", {})

    return ThresholdsConfig(
        dedup=DedupThresholds(
            title_exact_en=d.get("title_exact_en", 0.85),
            title_exact_zh=d.get("title_exact_zh", 0.70),
            title_fuzzy_low=d.get("title_fuzzy_low", 0.35),
            body_jaccard=d.get("body_jaccard", 0.50),
            entity_body_jaccard=d.get("entity_body_jaccard", 0.40),
            lookback_days=d.get("lookback_days", 7),
        ),
        text_processing=TextProcessingThresholds(
            summary_max_chars=tp.get("summary_max_chars", 500),
            min_sentence_len=tp.get("min_sentence_len", 15),
        ),
        filtering=FilteringThresholds(
            recency_windows_hours=tuple(f.get("recency_windows_hours", [72, 168])),
            min_signals=f.get("min_signals", 10),
            max_signals=f.get("max_signals", 75),
            max_per_source=f.get("max_per_source", 3),
        ),
        translation=TranslationThresholds(
            body_truncate_chars=tr.get("body_truncate_chars", 500),
            english_fragment_threshold=tr.get("english_fragment_threshold", 0.15),
        ),
    )


def _load_templates(config_dir: Path) -> TemplateData:
    """Load implication and perspective templates."""
    templates_dir = config_dir / "templates"

    impl = _load_yaml_optional(templates_dir / "implications.yaml")
    persp = _load_yaml_optional(templates_dir / "perspectives.yaml")

    return TemplateData(
        impact_templates=impl.get("impact_templates", {}),
        watch_templates=impl.get("watch_templates", {}),
        canada_perspective=persp.get("canada_perspective", {}),
        china_perspective=persp.get("china_perspective", {}),
    )


def _load_chinese_sources(config_dir: Path) -> ChineseSourceData:
    """Load Chinese source detection data."""
    data = _load_yaml_optional(config_dir / "chinese_sources.yaml")
    return ChineseSourceData(
        source_names=frozenset(data.get("source_names", [])),
        domains=frozenset(data.get("domains", [])),
        name_translations=data.get("name_translations", {}),
    )


def _load_relevance(config_dir: Path) -> RelevanceData:
    """Load relevance and value-scoring keyword lists."""
    data = _load_yaml_optional(config_dir / "relevance_keywords.yaml")
    return RelevanceData(
        china_relevance=data.get("china_relevance_keywords", []),
        low_value_patterns=data.get("low_value_patterns", []),
        high_value_keywords=data.get("high_value_keywords", []),
        canada_keywords=data.get("canada_keywords", []),
        china_keywords=data.get("china_keywords", []),
        canadian_sources=frozenset(data.get("canadian_sources", [])),
        regulatory_keywords=data.get("regulatory_keywords", []),
    )


def _load_text_patterns(config_dir: Path) -> TextPatternData:
    """Load text processing pattern lists."""
    data = _load_yaml_optional(config_dir / "text_processing.yaml")
    return TextPatternData(
        filler_patterns=data.get("filler_patterns", []),
        key_point_patterns=data.get("key_point_patterns", []),
        boilerplate_patterns=data.get("boilerplate_patterns", []),
    )


def load_config(
    env: str | None = None,
    config_dir: Path | None = None,
) -> AppConfig:
    """Load and parse the YAML config for the given environment.

    Args:
        env: The environment name (dev/staging/prod). Auto-detected if None.
        config_dir: Override the config directory path.

    Returns:
        Fully resolved AppConfig instance.
    """
    resolved_env = detect_env(env)
    resolved_config_dir = config_dir or PROJECT_ROOT / "config"
    config_path = resolved_config_dir / f"analysis.{resolved_env}.yaml"

    raw = _load_yaml(config_path)

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        raw_dir=paths_raw.get("raw_dir", "../cc-data/raw"),
        processed_dir=paths_raw.get("processed_dir", "../cc-data/processed"),
        archive_dir=paths_raw.get("archive_dir", "../cc-data/archive"),
        schemas_dir=paths_raw.get("schemas_dir", "../cc-data/schemas"),
    )

    tension_raw = raw.get("tension_index", {})
    tension = TensionConfig(
        window_days=tension_raw.get("window_days", 30),
        cap_denominator=tension_raw.get("cap_denominator", 20),
    )

    logging_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=logging_raw.get("level", "INFO"),
        format=logging_raw.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )

    validation_raw = raw.get("validation", {})
    validation = ValidationConfig(
        strict=validation_raw.get("strict", True),
        schema_file=validation_raw.get("schema_file", "briefing.schema.json"),
    )

    keywords = _load_keyword_dicts(resolved_config_dir)
    thresholds = _load_thresholds(raw)
    templates = _load_templates(resolved_config_dir)
    chinese_sources = _load_chinese_sources(resolved_config_dir)
    relevance = _load_relevance(resolved_config_dir)
    text_patterns = _load_text_patterns(resolved_config_dir)

    return AppConfig(
        env=resolved_env,
        paths=paths,
        tension=tension,
        logging=logging_cfg,
        validation=validation,
        keywords=keywords,
        thresholds=thresholds,
        templates=templates,
        chinese_sources=chinese_sources,
        relevance=relevance,
        text_patterns=text_patterns,
    )
