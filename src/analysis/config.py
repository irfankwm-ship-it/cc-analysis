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


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    env: str
    paths: PathsConfig
    tension: TensionConfig
    logging: LoggingConfig
    validation: ValidationConfig
    keywords: KeywordDicts


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

    return AppConfig(
        env=resolved_env,
        paths=paths,
        tension=tension,
        logging=logging_cfg,
        validation=validation,
        keywords=keywords,
    )
