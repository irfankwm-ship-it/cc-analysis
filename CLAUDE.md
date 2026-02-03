# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**cc-analysis** is a deterministic, rule-based analysis engine for the China Compass pipeline. It transforms raw signal data into structured briefings with classified signals, a composite tension index, entity tracking, and situation monitoring. Zero LLM inference — every classification is keyword-driven or numeric for reproducibility.

## Architecture

### Pipeline Steps (in order)

1. **Load raw signals** from `cc-data/raw/{date}/` — handles fetcher envelope format, extracts nested arrays
2. **Load supplementary data** — trade (statcan), market (yahoo_finance), parliament
3. **Filter & prioritize** — recency gate (adaptive 72h–168h window until ≥10 signals), China-relevance check, bilateral prioritization, source diversification (round-robin)
4. **Deduplicate** — three-tier: URL exact match → title similarity ≥0.80 → title 0.50–0.80 AND body Jaccard ≥0.60
5. **Classify** — source tier mapping → category (8 categories via keyword scoring) → severity (multi-factor score)
6. **Compute trends** — day-over-day comparison from previous briefing
7. **Compute tension index** — weighted composite (0–10 scale) across 6 components
8. **Match entities** — scan signals against `entity_aliases.yaml` dictionary (21 entities)
9. **Track active situations** — trigger keyword matching against 6 known situations
10. **Generate supplementary content** — volume number, today's number, quote of the day
11. **Assemble & validate** — JSON Schema validation with local `$ref` resolution
12. **Write output** — `processed/{date}/briefing.json` + `processed/latest/` + `archive/daily/{date}/`

### Classification Details

**Category scoring**: concatenates all text fields (title, body, headline, summary, etc.) in both EN and ZH. Per-category keyword match: exact word = +3, phrase match = +3, partial (>2 chars) = +1. Tiebreaker: specificity order (legal > social > economic > political > military > technology > diplomatic > trade).

**Severity scoring** (4 factors, summed, floored at 0):
- Source tier: official=4, wire=3, specialist=2, media=1
- Keyword modifiers: escalation=+3, moderate_escalation=+2, de-escalation=-2
- Bilateral directness: direct Canada-China=+2, general China=+1
- Recency: same day=+1, within 7 days=0, older=-1

Score thresholds: ≥8 critical, ≥6 high, ≥4 elevated, ≥2 moderate, ≥0 low.

### Tension Index

6 components with weights (must sum to 1.0): Diplomatic 0.25, Trade 0.25, Military 0.15, Political 0.15, Technology 0.10, Social 0.10. Economic and Legal categories exist but are excluded from the tension index.

Component score: `min(raw_severity_points / cap_denominator * 10, 10)`, rounded to integer. Composite: weighted sum, rounded to 1 decimal.

Level thresholds: ≥9.0 Critical, ≥7.0 High, ≥4.1 Elevated, ≥2.1 Moderate, ≥0.0 Low.

## Build & Development Commands

```bash
poetry install                                    # install deps
poetry run pytest                                 # all tests
poetry run pytest tests/test_tension_index.py     # single test file
poetry run pytest -k "severity_upgrade"           # pattern match
poetry run ruff check src/ tests/                 # lint
poetry run analysis run --env dev                 # full pipeline
poetry run analysis run --env prod --date 2025-01-30  # specific date
poetry run analysis compile-volume --date 2025-02-01  # compile previous month
```

## CLI Reference

```
analysis run [--env dev|staging|prod] [--date YYYY-MM-DD] [--raw-dir DIR] [--output-dir DIR] [--archive-dir DIR] [--schemas-dir DIR]
analysis compile-volume [--env dev|staging|prod] [--date YYYY-MM-DD] [--archive-dir DIR]
```

## Configuration

Config files: `config/analysis.{env}.yaml`. Environment resolved via: `--env` → `CC_ENV` → default `dev`.

Keyword dictionaries in `config/keyword_dicts/`:
- `categories.yaml` — 8 category keyword lists (EN + ZH)
- `severity_modifiers.yaml` — escalation (+3), moderate_escalation (+2), de-escalation (-2) keywords
- `entity_aliases.yaml` — 21 entities with bilingual alias lists

Config hierarchy: `AppConfig` → `PathsConfig`, `TensionConfig`, `LoggingConfig`, `ValidationConfig`, `KeywordDicts`.

## Key Conventions

- Keyword matching is case-insensitive for English, exact substring for Chinese
- All processed text fields use bilingual objects `{"en": "...", "zh": "..."}`
- Component scores are integers; composite uses unrounded floats before final 1-decimal rounding
- Dedup stop words exclude common EN words from Jaccard body comparison
- Translation uses MyMemory free API; gracefully returns original text on failure
- Empty raw directory is handled gracefully (empty signal list with warning)
- If strict validation is on and fails, raises `ClickException` (non-zero exit)
- Ruff config: line-length 100, target Python 3.12, rules E/F/I/N/W/UP

## Extension Points

- **New category**: add to `categories.yaml` + `VALID_CATEGORIES` + `SPECIFICITY_ORDER`; optionally add to `COMPONENT_WEIGHTS` in `tension_index.py` (rebalance to sum 1.0)
- **New entity**: add to `entity_aliases.yaml`; optionally map type in `entity_types` dict in `entities.py`
- **New situation**: add to `KNOWN_SITUATIONS` in `active_situations.py` (auto-detection, no other changes)
- **Adjust severity**: edit `SEVERITY_THRESHOLDS` in `severity.py` and/or keyword weights in `severity_modifiers.yaml`
