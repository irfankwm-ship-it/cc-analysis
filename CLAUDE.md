# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**cc-analysis** is a hybrid rule-based + LLM analysis engine for the China Compass pipeline. It transforms raw signal data into structured briefings with classified signals, a composite tension index, entity tracking, and situation monitoring. Classification is keyword-driven for reproducibility; LLM (local Qwen2.5 via ollama) handles translation and summarization of high-priority signals.

## Architecture

### Module Structure

```
src/analysis/
├── cli.py                  # CLI entry point (~480 lines) — Click commands + orchestration
├── config.py               # Config loading with frozen dataclasses + YAML loaders
├── signal_types.py         # TypedDict definitions: RawSignal, ClassifiedSignal, NormalizedSignal
├── signal_filtering.py     # Load, filter, prioritize raw signals (recency, relevance, diversity)
├── signal_normalization.py # Bilingual conversion, implications, perspectives, LLM summarization
├── text_processing.py      # Sentence splitting, scoring, summarization, boilerplate removal
├── source_detection.py     # Chinese source identification and name translation
├── data_transforms.py      # Market/trade/parliament data transforms, volume numbers, quotes
├── translate.py            # Concurrent LLM + MyMemory translation with quality validation
├── output.py               # Briefing assembly, JSON Schema validation, file writing
├── timeline_compiler.py    # Timeline aggregation, dedup, milestone management
├── dedup.py                # 4-tier signal deduplication with 7-day lookback
├── entities.py             # Entity matching against entity_aliases.yaml (21 entities)
├── active_situations.py    # Situation tracking by keyword matching (6 known situations)
├── llm.py                  # LLM calls (ollama): translate, summarize, perspectives
├── trend.py                # Day-over-day signal comparison
├── tension_index.py        # 6-component weighted tension index (0–10 scale)
├── volume_compiler.py      # Monthly volume aggregation
└── classifiers/
    ├── category.py         # 8-category keyword scoring
    ├── severity.py         # 4-factor severity scoring
    └── source_mapper.py    # Source tier classification (official/wire/specialist/media)
```

### Pipeline Steps (in order)

1. **Load raw signals** (`signal_filtering.load_raw_signals`) — handles fetcher envelope format, extracts nested arrays
2. **Load supplementary data** (`data_transforms.load_supplementary_data`) — trade (statcan), market (yahoo_finance), parliament
3. **Filter & prioritize** (`signal_filtering.filter_and_prioritize_signals`) — recency gate (adaptive 72h–168h window until ≥10 signals), China-relevance check, bilateral prioritization, source diversification (round-robin)
4. **Pre-classify** — add category and entity_ids to signals before dedup (enables entity-based dedup)
5. **Deduplicate** (`dedup`) — four-tier with 7-day lookback:
   - Tier 1: URL exact match
   - Tier 2: Title similarity ≥0.80 (EN) or ≥0.70 (ZH — shorter headlines)
   - Tier 3: Title 0.50–0.80 AND body Jaccard ≥0.60
   - Tier 4: Same entities + same category + body Jaccard ≥0.50 (catches same story from different sources)
6. **Classify** (`classifiers/`) — source tier mapping → category (8 categories via keyword scoring) → severity (multi-factor score)
7. **Normalize & summarize** (`signal_normalization.normalize_signal`) — convert to bilingual schema; LLM summarization for critical/high severity signals; ensure complete sentences (no truncation)
8. **Translate** (`translate.translate_to_chinese/translate_to_english`) — concurrent LLM translation with strict mode retry, MyMemory API fallback, dictionary-based cleanup
9. **Generate perspectives** (`signal_normalization.generate_perspectives`) — LLM-powered dual perspectives (Canada/Beijing viewpoints) with template fallback
10. **Compute trends** (`trend`) — day-over-day comparison from previous briefing
11. **Compute tension index** (`tension_index`) — weighted composite (0–10 scale) across 6 components
12. **Match entities** (`entities`) — scan signals against `entity_aliases.yaml` dictionary (21 entities)
13. **Track active situations** (`active_situations`) — trigger keyword matching against 6 known situations
14. **Generate supplementary content** (`data_transforms`) — volume number, today's number, quote of the day
15. **Assemble & validate** (`output.assemble_briefing`, `output.validate_briefing`) — JSON Schema validation with local `$ref` resolution
16. **Write output** (`output.write_processed`, `output.write_archive`) — `processed/{date}/briefing.json` + `processed/latest/` + `archive/daily/{date}/`

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
poetry run pytest                                 # all 355 tests
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
analysis compile-timeline [--env dev|staging|prod] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--archive-dir DIR] [--timelines-dir DIR]
analysis mark-milestone SIGNAL_ID [--timeline-category CATEGORY] [--archive-dir DIR]
```

### Timeline Commands

- `compile-timeline`: Aggregates daily briefings into `timelines/canada-china.json`. Extracts high-severity signals, tension trend data, and milestone events.
- `mark-milestone`: Flags a signal as historically significant for timeline inclusion. Categories: crisis, escalation, de-escalation, agreement, policy_shift, leadership, incident, sanction, negotiation.

## Configuration

### Environment Config

Config files: `config/analysis.{env}.yaml`. Environment resolved via: `--env` → `CC_ENV` → default `dev`.

Each env config includes a `thresholds:` section with tunable values for dedup, text processing, filtering, and translation.

### Data Config (YAML)

Externalized data that was previously hardcoded:

| File | Contents |
|------|----------|
| `config/templates/implications.yaml` | Impact templates + watch templates (8 categories × en/zh) |
| `config/templates/perspectives.yaml` | Canada + Beijing perspective templates (8 categories × en/zh) |
| `config/chinese_sources.yaml` | Source names (54), domains (19), name translations (30+) |
| `config/relevance_keywords.yaml` | China relevance, low-value patterns, high-value keywords, Canada/China keywords |
| `config/text_processing.yaml` | Filler patterns, key-point patterns, boilerplate patterns |

### Keyword Dictionaries

`config/keyword_dicts/`:
- `categories.yaml` — 8 category keyword lists (EN + ZH)
- `severity_modifiers.yaml` — escalation (+3), moderate_escalation (+2), de-escalation (-2) keywords
- `entity_aliases.yaml` — 21 entities with bilingual alias lists

### Config Hierarchy

`AppConfig` → `PathsConfig`, `TensionConfig`, `LoggingConfig`, `ValidationConfig`, `KeywordDicts`, `ThresholdsConfig`, `TemplateData`, `ChineseSourceData`, `RelevanceData`, `TextPatternData`.

All config dataclasses are frozen. New sections have defaults matching previous hardcoded values, so missing YAML keys don't break anything.

## Key Conventions

- Keyword matching is case-insensitive for English, exact substring for Chinese
- All processed text fields use bilingual objects `{"en": "...", "zh": "..."}`
- Component scores are integers; composite uses unrounded floats before final 1-decimal rounding
- Dedup stop words exclude common EN words AND Chinese function words (的, 了, 是, etc.) from Jaccard comparison
- Chinese text tokenized by character (no word boundaries); English by 3+ char words
- Empty raw directory is handled gracefully (empty signal list with warning)
- If strict validation is on and fails, raises `ClickException` (non-zero exit)
- Ruff config: line-length 100, target Python 3.12, rules E/F/I/N/W/UP

### Signal Types

`signal_types.py` defines TypedDict types for the three signal stages:
- `RawSignal` — as loaded from fetcher JSON (title may be str or bilingual dict)
- `ClassifiedSignal` — after category/severity/entity classification
- `NormalizedSignal` — final bilingual schema with implications and perspectives

These are purely additive type hints — all signal data still flows as plain dicts.

### Translation

- **Concurrent**: `ThreadPoolExecutor` with `CC_TRANSLATE_WORKERS` env var (default 3)
- **Pipeline**: LLM primary (local ollama) → strict mode retry → MyMemory fallback → dictionary cleanup
- **Quality checks**: `_contains_untranslated_english()` detects fragments, triggers strict retry
- **Cleanup**: `_clean_partial_translation()` handles "word（翻译）" patterns + `_UNTRANSLATED_WORDS` dict (170+ entries)
- **Gender pronouns**: `fix_gender_pronouns()` corrects he/his/him for 70+ known female figures
- **Semaphores**: `_OLLAMA_SEMAPHORE` (N workers) for LLM; `_MYMEMORY_LOCK` (1) for rate-limited API
- LLM config via env vars: `OLLAMA_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL` (default: qwen2.5:3b-instruct-q4_K_M)

### Text Processing

`text_processing.py` handles extractive summarization:
- `split_sentences()` — regex-based sentence splitting for EN and ZH
- `score_sentence()` — title-overlap, position, numbers, filler penalty
- `summarize_body()` — selects top sentences by score, falls back to LLM for critical/high severity
- `ensure_complete_sentences()` — trims to last complete sentence (no truncation artifacts)
- All functions accept optional pattern/threshold params; fall back to config-loaded defaults

### Signal Normalization

`signal_normalization.py` converts classified signals to final bilingual schema:
- `normalize_signal()` — main entry: bilingual conversion + implications + quote extraction
- `generate_implications()` — rule-based from category/severity using YAML templates
- `generate_perspectives()` — LLM-powered dual Canada/Beijing viewpoints with template fallback
- `translate_signals_batch()` — batch translation with preserved Chinese content for Chinese sources

### Chinese Source Detection

`source_detection.py` identifies Chinese-language sources:
- Source name set (54 names) and domain set (19 domains) loaded from `config/chinese_sources.yaml`
- `translate_source_name()` provides English names for Chinese sources
- Chinese sources get `original_zh_url` field for "view original" links

### Timeline Support

Signals can be marked for timeline inclusion with these optional fields:
- `is_milestone: true` — Flag historically significant events
- `timeline_category` — Event type: crisis, escalation, de-escalation, agreement, policy_shift, leadership, incident, sanction, negotiation
- `related_signals` — Array of related signal IDs for event chains

Active situations also support timeline tracking:
- `event_start_date` / `event_end_date` — Duration tracking for multi-day events
- `timeline_id` — Unique ID for linking across daily briefings

### Timeline Validation & Deduplication

The timeline compiler (`timeline_compiler.py`) enforces quality:
- **Translation validation**: Events must have proper Chinese text in `zh` field (not English or French)
- **CJK character check**: `_is_chinese_text()` verifies Chinese characters present
- **French detection**: `_is_likely_french()` catches untranslated French sources
- **Deduplication**: Same date + title similarity ≥0.70 = duplicate, keeps first occurrence
- **Existing event filter**: Invalid translations removed when loading existing timeline

## Testing

355 tests across 20 test files. Key test files for the extracted modules:

| Test File | Module | Tests |
|-----------|--------|-------|
| `test_text_processing.py` | text_processing.py | Sentence splitting, scoring, summarization |
| `test_source_detection.py` | source_detection.py | Chinese source ID, name translation |
| `test_signal_filtering.py` | signal_filtering.py | Relevance, value scoring, bilateral detection |
| `test_signal_normalization.py` | signal_normalization.py | Bilingual conversion, implications, perspectives |
| `test_data_transforms.py` | data_transforms.py | Market/trade/parliament transforms |
| `test_config.py` | config.py | Config loading, thresholds, templates |
| `test_output.py` | output.py | Briefing assembly, validation, file writing |
| `test_timeline_compiler.py` | timeline_compiler.py | Timeline compilation, dedup, milestones |
| `test_translate.py` | translate.py | Translation pipeline, gender pronouns, cleanup |
| `test_cli.py` | cli.py | End-to-end pipeline integration |

## Extension Points

- **New category**: add to `categories.yaml` + `VALID_CATEGORIES` + `SPECIFICITY_ORDER`; optionally add to `COMPONENT_WEIGHTS` in `tension_index.py` (rebalance to sum 1.0)
- **New entity**: add to `entity_aliases.yaml`; optionally map type in `entity_types` dict in `entities.py`
- **New situation**: add to `KNOWN_SITUATIONS` in `active_situations.py` (auto-detection, no other changes)
- **Adjust severity**: edit `SEVERITY_THRESHOLDS` in `severity.py` and/or keyword weights in `severity_modifiers.yaml`
- **Adjust thresholds**: edit `thresholds:` section in `config/analysis.{env}.yaml`
- **Add template data**: edit YAML files in `config/templates/` or `config/` — no code changes needed
- **Translation concurrency**: set `CC_TRANSLATE_WORKERS` env var (default 3)
