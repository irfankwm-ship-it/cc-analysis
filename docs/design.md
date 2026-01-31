# cc-analysis Design Document

## 1. Overview

### Purpose

`cc-analysis` is the rule-based analysis engine for the **China Compass** pipeline. It consumes raw signal data (news articles, government statements, market data) fetched by upstream collectors, classifies each signal by category and severity, computes a composite tension index, tracks ongoing situations, resolves entity mentions, and emits a structured `briefing.json` envelope that downstream renderers consume to produce the daily bilingual (EN/ZH) briefing.

### No-LLM Philosophy

Every classification, scoring, and aggregation step in `cc-analysis` is deterministic and rule-based. The engine relies exclusively on keyword dictionaries, numeric formulas, and threshold tables -- never on large language models or probabilistic inference. This guarantees:

- **Reproducibility**: identical inputs always produce identical outputs.
- **Auditability**: every score can be traced back to specific keyword matches, source tier lookups, and weight constants.
- **Cost and latency**: the pipeline runs in milliseconds with zero API calls.
- **Offline operation**: no network access is required at analysis time.

### Role in Pipeline

```
cc-fetchers  -->  cc-data/raw/{date}/*.json
                        |
                  cc-analysis (this repo)
                        |
                  cc-data/processed/{date}/briefing.json
                  cc-data/archive/daily/{date}/briefing.json
                        |
                  cc-site (renderer)
```

`cc-analysis` sits between the fetcher layer and the presentation layer. It reads raw JSON files deposited by fetchers, applies the full classification and scoring pipeline, and writes the canonical `briefing.json` that the site generator consumes.

---

## 2. Architecture

### Module Map

```
src/analysis/
  __init__.py                # Package version (0.1.0)
  cli.py                     # Click CLI: `analysis run` and `analysis compile-volume`
  config.py                  # Environment-aware YAML config loader
  classifiers/
    __init__.py
    source_mapper.py          # Source name -> reliability tier
    category.py               # Keyword-based category classification
    severity.py               # Multi-factor severity scoring
  tension_index.py            # Weighted composite tension index
  trend.py                    # Day-over-day comparison
  entities.py                 # Dictionary-based entity matching
  active_situations.py        # Ongoing situation tracker
  volume_compiler.py          # Monthly volume aggregation
  output.py                   # Briefing assembly, schema validation, file I/O

config/
  analysis.dev.yaml           # Dev environment config
  analysis.staging.yaml       # Staging environment config
  analysis.prod.yaml          # Production environment config
  keyword_dicts/
    categories.yaml           # 8-category keyword dictionaries (EN + ZH)
    severity_modifiers.yaml   # Escalation / de-escalation keyword lists with weights
    entity_aliases.yaml       # Entity canonical IDs to alias mappings
```

### CLI

The CLI is built with Click and exposes two commands via the `analysis` entry point (registered in `pyproject.toml` as `analysis = "analysis.cli:main"`):

| Command | Description |
|---|---|
| `analysis run` | Execute the full daily analysis pipeline for a given date. |
| `analysis compile-volume` | Aggregate daily briefings from the previous month into a volume summary. |

**`analysis run` options:**

| Flag | Default | Purpose |
|---|---|---|
| `--env` | `dev` (or `CC_ENV`) | Environment selector: `dev`, `staging`, `prod`. |
| `--date` | today | Target analysis date (`YYYY-MM-DD`). |
| `--raw-dir` | `../cc-data/raw/{date}/` | Directory containing raw fetcher output. |
| `--output-dir` | `../cc-data/processed/` | Processed output directory. |
| `--archive-dir` | `../cc-data/archive/` | Archive directory for daily storage. |
| `--schemas-dir` | `../cc-data/schemas/` | JSON Schema directory for validation. |

**`analysis compile-volume` options:**

| Flag | Default | Purpose |
|---|---|---|
| `--env` | `dev` (or `CC_ENV`) | Environment selector. |
| `--date` | today | Reference date; the previous month is compiled. |
| `--archive-dir` | `../cc-data/archive/` | Archive directory containing daily briefings. |

### Pipeline Execution Order (`analysis run`)

```
 1. Load raw signals from JSON files in the raw directory
 2. Load supplementary data (trade, market, parliament)
 3. Classify each signal:
    a. Map source to reliability tier
    b. Classify category via keyword scoring
    c. Classify severity via multi-factor scoring
 4. Compute day-over-day trends from previous briefing
 5. Compute the composite tension index
 6. Match entities across all signals
 7. Track active situations
 8. Determine volume number from archive history
 9. Assemble the briefing envelope
10. Validate against JSON Schema
11. Write to processed and archive directories
```

### Raw Signal Loading

The loader (`_load_raw_signals`) handles several payload shapes from upstream fetchers:

- **Fetcher envelope format**: `{"metadata": {...}, "data": {...}}` -- extracts from `data`.
- **Nested arrays**: looks for `signals`, `articles`, `items`, or `results` keys within the payload.
- **Bare lists**: treated directly as signal arrays.
- **Single objects**: if a dict has a `title` or `headline` field, treated as a single signal.

Supplementary data files are loaded by filename convention: `statcan.json` or `trade.json` for trade data, `yahoo_finance.json` or `market.json` for market data, and `parliament.json` for parliamentary data.

---

## 3. Classification Pipeline

Raw signals pass through a three-stage classification pipeline. Each stage is pure -- it reads input data and keyword dictionaries and returns a deterministic result.

### Stage 1: Source Mapping

**Module:** `src/analysis/classifiers/source_mapper.py`

Maps the signal's `source` field to one of four reliability tiers:

| Tier | Score | Examples (EN) |
|---|---|---|
| `official` | 4 | Global Affairs Canada, PMO, State Council, MOFCOM, PBOC, MFA, Taiwan Ministry of National Defense |
| `wire` | 3 | Reuters, AP, AFP, Bloomberg |
| `specialist` | 2 | CSIS, Sinocism, China Brief, MERICS, OSINT |
| `media` | 1 | Globe and Mail, CBC, South China Morning Post, Xinhua, CAC, SAMR |

Each tier also includes Chinese-language source names for bilingual matching.

**Matching strategy:**

1. Build a flat lookup table at import time: `{source_name_lower: tier}` for all known EN and ZH source names.
2. **Exact match**: case-insensitive lookup against the flat table.
3. **Substring match** (if exact match fails): bidirectional -- checks if any known source name is contained in the input or vice versa.
4. **Fallback**: unknown sources default to `"media"` (tier score 1).
5. **Bilingual handling**: for dict-typed source fields (`{"en": "...", "zh": "..."}`), tries English first; if the result is `"media"` and a Chinese name is present, retries with the Chinese name.

### Stage 2: Category Classification

**Module:** `src/analysis/classifiers/category.py`

Classifies each signal into one of eight categories using keyword scoring.

**Categories:** `diplomatic`, `trade`, `military`, `technology`, `political`, `economic`, `social`, `legal`

**Text extraction:** the classifier concatenates text from `title`, `body`, `headline`, `summary`, `content`, and `description` fields, handling both string and bilingual dict (`{"en": "...", "zh": "..."}`) formats.

**Scoring algorithm:**

1. For each category, score the concatenated text against both its EN and ZH keyword lists.
2. Scoring rules:
   - **Exact word match** (single-word keyword found as a discrete word in the text's word set): **+3 points**.
   - **Phrase match** (multi-word keyword found as substring in the text): **+3 points**.
   - **Partial match** (any word fragment longer than 2 characters from the keyword found as substring): **+1 point**.
3. Sum EN + ZH scores per category.
4. Select the category with the highest total score.
5. **Tiebreaker:** when multiple categories share the max score, prefer the more specific category according to a fixed specificity order:

```
legal > social > economic > political > military > technology > diplomatic > trade
```

6. **Default:** if no keywords match (all scores zero), defaults to `"political"`.

### Stage 3: Severity Classification

**Module:** `src/analysis/classifiers/severity.py`

Computes a multi-factor severity score and maps it to a severity level.

**Four scoring factors:**

| Factor | Range | Description |
|---|---|---|
| Source tier | 1--4 | From source mapping (`official`=4, `wire`=3, `specialist`=2, `media`=1). |
| Keyword modifiers | -2 to +3 | `escalation` (+3), `moderate_escalation` (+2), `de_escalation` (-2). |
| Bilateral directness | 0--2 | 2 if Canada-China directly mentioned, 1 if China generally mentioned, 0 otherwise. |
| Recency | -1 to +1 | +1 for same-day, 0 for within 7 days, -1 for older than 7 days. |

**Raw score** = sum of all four factors, floored at 0 (never negative).

**Severity thresholds:**

| Score | Level |
|---|---|
| >= 10 | `critical` |
| >= 7 | `high` |
| >= 5 | `elevated` |
| >= 3 | `moderate` |
| >= 0 | `low` |

**Keyword modifier matching:** each modifier group (`escalation`, `moderate_escalation`, `de_escalation`) is checked independently. For each group, if any keyword (EN checked first, then ZH) matches as a substring, the group's `weight` is added to the score. At most one match per group is counted. A signal mentioning both escalation and de-escalation terms receives both the +3 and -2 modifiers (net +1).

**Bilateral keywords (hardcoded):**
- Direct Canada-China (score +2): `canada-china`, `canada china`, `sino-canadian`, `canadian`, `ottawa`, `beijing`, `trudeau`, `canada`, plus Chinese equivalents.
- General China (score +1): `china`, `chinese`, `beijing`, `prc`, plus Chinese equivalents.

**Recency scoring** supports multiple date formats: `YYYY-MM-DD`, `Month DD, YYYY`, `DD Month YYYY`, `YYYY/MM/DD`. Unparseable dates receive a neutral score of 0.

---

## 4. Tension Index

**Module:** `src/analysis/tension_index.py`

The tension index is a weighted composite score on a 0--10 scale that summarizes the overall state of Canada-China relations.

### Components and Weights

Six of the eight categories are tracked as tension index components:

| Component | Weight |
|---|---|
| Diplomatic | 0.25 |
| Trade | 0.25 |
| Military | 0.15 |
| Political | 0.15 |
| Technology | 0.10 |
| Social | 0.10 |
| **Total** | **1.00** |

Note: `economic` and `legal` categories contribute to signal classification but are not tracked as separate tension index components.

### Severity Points

Each severity level maps to a point value used in tension index computation:

| Severity | Points |
|---|---|
| critical | 5 |
| high | 4 |
| elevated | 3 |
| moderate | 2 |
| low | 1 |

### Formula

For each component category `c`:

```
raw_points[c] = sum(SEVERITY_POINTS[s.severity] for s in signals where s.category == c)
component_score[c] = min(raw_points[c] / cap_denominator * 10, 10)
```

The `cap_denominator` is configurable (default: **20**). It controls how quickly a component saturates to its maximum of 10. For example, with `cap_denominator=20`, a category needs 20 severity points (e.g., 4 critical signals) to reach a score of 10.

Component scores are rounded to the nearest integer for the output schema.

**Composite score:**

```
composite = round(sum(component_score[c] * weight[c] for c in all_components), 1)
```

The composite uses the unrounded floating-point component scores for precision, then rounds the final result to one decimal place.

### Level Mapping

| Composite | Level (EN) | Level (ZH) |
|---|---|---|
| >= 9.0 | Critical | (Chinese equivalent) |
| >= 7.0 | High | (Chinese equivalent) |
| >= 4.1 | Elevated | (Chinese equivalent) |
| >= 2.1 | Moderate | (Chinese equivalent) |
| >= 0.0 | Low | (Chinese equivalent) |

### Delta and Trend Computation

- **Delta:** `composite_today - composite_yesterday`, rounded to one decimal place. Zero if no previous day data exists.
- **Delta description:** bilingual text like `"+0.3 from previous day"` / `"No change from previous day"`.
- **Component trend:** each component compares its integer score to the previous day's score and reports `"up"`, `"down"`, or `"stable"`.
- **Key driver:** for each component, the title of the highest-severity signal in that category is selected. If no signals exist for a component, the key driver is `"No significant activity"`.

### Data Classes

- `ComponentScore`: holds `name` (bilingual dict), `score` (int 0--10), `weight` (float), `trend` (string), `key_driver` (bilingual dict).
- `TensionIndex`: holds `composite` (float), `level` (bilingual dict), `delta` (float), `delta_description` (bilingual dict), `components` (list of `ComponentScore`). Both provide `.to_dict()` for serialization into the briefing envelope.

---

## 5. Entity Matching

**Module:** `src/analysis/entities.py`

### Dictionary-Based Approach

Entities are matched using the `entity_aliases.yaml` dictionary. Each entry maps a canonical entity ID (snake_case) to lists of English and Chinese aliases:

```yaml
xi_jinping:
  en:
    - Xi Jinping
    - Xi
    - President Xi
    - General Secretary Xi
  zh:
    - (Chinese equivalents)
```

Currently tracked entities (12 total):

| ID | Type | Description |
|---|---|---|
| `xi_jinping` | people | Chinese head of state |
| `wang_yi` | people | Chinese foreign minister |
| `two_michaels` | people | Michael Kovrig and Michael Spavor |
| `mofcom` | institution | Ministry of Commerce |
| `mfa` | institution | Ministry of Foreign Affairs |
| `csis` | institution | Canadian Security Intelligence Service |
| `ufwd` | institution | United Front Work Department |
| `mss` | institution | Ministry of State Security |
| `huawei` | org | Technology company |
| `canola` | commodity | Agricultural export commodity |
| `rare_earths` | commodity | Critical minerals |
| `softwood_lumber` | commodity | Forestry export commodity |

### Alias Resolution

For each signal, the matcher:

1. Extracts all searchable text from `title`, `body`, `headline`, `summary`, `content`, `description`, and nested `implications.canada_impact` / `implications.what_to_watch` fields (both EN and ZH variants).
2. For each entity, checks EN aliases via case-insensitive substring match against the lowercased text.
3. If no EN alias matches, checks ZH aliases via exact substring match against the original text (Chinese text is not lowercased).
4. Short-circuits on first matching alias per entity per language -- finding one alias is sufficient to confirm the entity's presence.
5. Returns a deduplicated, sorted list of matched entity IDs per signal.

### Aggregation

`match_entities_across_signals` counts how many signals mention each entity. Results are sorted by mention count (descending).

`build_entity_directory` constructs schema-conformant entity objects with:

- `id`: canonical entity ID.
- `name`: bilingual dict using the first alias in each language list as the primary name.
- `type`: one of `people`, `institution`, `org`, or `commodity` (mapped from a hardcoded lookup by entity ID, defaulting to `org` for unmapped IDs).
- `description`: bilingual text indicating mention count (e.g., "Mentioned in 3 signal(s) today.").
- `has_detail_page`: always `false`.

---

## 6. Active Situations

**Module:** `src/analysis/active_situations.py`

### Known Situations

Six predefined situation patterns are tracked:

| ID | Name | Default Severity | Start Date |
|---|---|---|---|
| `canola_trade_dispute` | Canola Trade Dispute | elevated | 2019-03-01 |
| `tech_decoupling` | Tech Decoupling | high | 2018-12-01 |
| `foreign_interference` | Foreign Interference Investigation | high | 2023-02-01 |
| `taiwan_strait_tensions` | Taiwan Strait Tensions | elevated | 2022-08-01 |
| `rare_earth_controls` | Rare Earth Export Controls | elevated | 2023-07-01 |
| `diplomatic_tensions` | Diplomatic Tensions | moderate | 2018-12-01 |

Each situation includes a bilingual name and a list of trigger keywords in both English and Chinese.

### Trigger Detection

A signal matches a situation if any trigger keyword appears (case-insensitive for English, exact substring for Chinese) in the signal's `title`, `body`, `headline`, or `summary` fields.

A situation is included in the output **only if at least one signal matches its trigger keywords on the current day**. A single signal can trigger multiple situations simultaneously.

### Day Counting

`day_count` is computed as the number of calendar days between the situation's predefined `start_date` and the current analysis date. This provides a running count of how long a situation has been active (e.g., the canola trade dispute may show 2000+ days).

### Severity Upgrade

Each situation starts at its `default_severity`. For every matching signal, the tracker compares the signal's classified severity to the situation's current severity. If the signal's severity is higher, the situation is upgraded. The ordering is:

```
critical (5) > high (4) > elevated (3) > moderate (2) > low (1)
```

This means a single critical-severity signal about canola will upgrade the Canola Trade Dispute from its default `elevated` to `critical`.

### Output

Active situations are serialized as:

```json
{
  "name": {"en": "Canola Trade Dispute", "zh": "..."},
  "detail": {"en": "2 related signal(s) detected today.", "zh": "..."},
  "severity": "high",
  "day_count": 2161
}
```

Results are sorted by severity (highest first). The `day_count` and `deadline` fields are omitted when zero or null respectively.

---

## 7. Volume Compilation

**Module:** `src/analysis/volume_compiler.py`

The `analysis compile-volume` command aggregates daily briefings from the previous calendar month into a monthly volume summary.

### Monthly Aggregation Logic

1. **Date range**: given a reference date, compute the previous month's first and last days. For example, reference date `2025-02-01` compiles January 2025 (2025-01-01 through 2025-01-31).
2. **Load briefings**: iterate through each day in the range and load `briefing.json` from `archive/daily/{date}/briefing.json` (with `archive/daily/{date}.json` as a flat-file fallback).
3. **Aggregate** across all loaded briefings:
   - **Total signal count**: sum of `len(briefing.signals)` across all days.
   - **Category breakdown**: count of signals per category (e.g., `{"diplomatic": 30, "trade": 25, ...}`).
   - **Severity breakdown**: count of signals per severity level (e.g., `{"critical": 10, "high": 40, ...}`).
   - **Tension trend**: list of `{"date": "YYYY-MM-DD", "value": <composite>}` pairs tracking daily composite scores.
4. **Volume numbering**: auto-incremented from existing `archive/volumes/vol-NNN.json` files.
5. **Output**: written to `archive/volumes/vol-{NNN}.json` (zero-padded to 3 digits).

### Volume Metadata Schema

```json
{
  "volume_number": 1,
  "period_start": "2025-01-01",
  "period_end": "2025-01-31",
  "signal_count": 150,
  "tension_trend": [
    {"date": "2025-01-01", "value": 5.2},
    {"date": "2025-01-02", "value": 5.4}
  ],
  "category_breakdown": {"diplomatic": 30, "trade": 25},
  "severity_breakdown": {"critical": 10, "high": 40}
}
```

---

## 8. Keyword Dictionaries

All keyword dictionaries live under `config/keyword_dicts/` as YAML files. They are loaded once at startup via the configuration system and passed into classifier functions.

### Structure

**`categories.yaml`** -- maps each of the 8 categories to bilingual keyword lists:

```yaml
diplomatic:
  en:
    - ambassador
    - embassy
    - consul
    - diplomatic
    - bilateral
    - foreign affairs
    - summoned
    - envoy
    # ... (14 EN keywords total)
  zh:
    - (12 ZH keywords)

trade:
  en:
    - tariff
    - trade
    - export
    - import
    # ... (15 EN keywords)
  zh:
    - (13 ZH keywords)
```

All eight categories follow this same structure. Keyword counts range from 11 to 15 per language per category.

**`severity_modifiers.yaml`** -- three modifier groups with weighted keyword lists:

```yaml
escalation:
  en:
    - detention
    - detained
    - arrested
    - sanctions
    - military confrontation
    - ban
    - crisis
    - emergency
    - war
    # ... (15 EN keywords)
  zh:
    - (12 ZH keywords)
  weight: 3

moderate_escalation:
  en:
    - tariff
    - restriction
    - investigation
    - probe
    - warning
    - dispute
    - tension
    # ... (10 EN keywords)
  zh:
    - (9 ZH keywords)
  weight: 2

de_escalation:
  en:
    - agreement
    - cooperation
    - dialogue
    - resumption
    - normalized
    - eased
    - lifted
    - resolved
  zh:
    - (8 ZH keywords)
  weight: -2
```

**`entity_aliases.yaml`** -- canonical entity IDs to bilingual alias lists:

```yaml
xi_jinping:
  en:
    - Xi Jinping
    - Xi
    - President Xi
    - General Secretary Xi
  zh:
    - (Chinese equivalents)

mofcom:
  en:
    - MOFCOM
    - Ministry of Commerce
  zh:
    - (Chinese equivalent)
```

Twelve entities are currently defined.

### Bilingual Support

Every keyword list includes both `en` and `zh` keys. English matching is case-insensitive (text and keywords are both lowercased). Chinese matching is exact substring (Chinese text is not case-folded). Both language keyword lists are scored or checked together during classification and entity matching.

### How to Extend

**Adding keywords to an existing category:**

1. Open `config/keyword_dicts/categories.yaml`.
2. Append the new keyword to the appropriate category's `en` or `zh` list.
3. Run the test suite (`pytest`) to verify no regressions in category classification.

**Adding a new severity modifier keyword:**

1. Open `config/keyword_dicts/severity_modifiers.yaml`.
2. Add the keyword to the appropriate group (`escalation`, `moderate_escalation`, or `de_escalation`) under `en` or `zh`.

**Adding a new entity:**

1. Open `config/keyword_dicts/entity_aliases.yaml`.
2. Add a new entry with a snake_case canonical ID and bilingual alias lists.
3. Optionally add a type mapping in `entity_types` in `src/analysis/entities.py`.

---

## 9. Configuration

**Module:** `src/analysis/config.py`

### Environment System

Three environments are supported: `dev`, `staging`, `prod`. Each has a corresponding YAML config file:

| File | Log Level | Strict Validation |
|---|---|---|
| `config/analysis.dev.yaml` | `DEBUG` | `true` |
| `config/analysis.staging.yaml` | `INFO` | `true` |
| `config/analysis.prod.yaml` | `WARNING` | `true` |

**Environment detection priority:**

1. Explicit `--env` CLI flag.
2. `CC_ENV` environment variable.
3. Default: `dev`.

Invalid environment names raise a `ValueError`.

### Configuration Structure

The `AppConfig` dataclass is the top-level configuration object, composed of frozen (`frozen=True`) sub-dataclasses:

```
AppConfig
  env: str                          # "dev" | "staging" | "prod"
  paths: PathsConfig
    raw_dir: str                    # Default: ../cc-data/raw
    processed_dir: str              # Default: ../cc-data/processed
    archive_dir: str                # Default: ../cc-data/archive
    schemas_dir: str                # Default: ../cc-data/schemas
  tension: TensionConfig
    window_days: int                # Default: 30
    cap_denominator: int            # Default: 20
  logging: LoggingConfig
    level: str                      # Default: INFO
    format: str                     # Default: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  validation: ValidationConfig
    strict: bool                    # Default: true
    schema_file: str                # Default: briefing.schema.json
  keywords: KeywordDicts
    categories: dict                # Loaded from categories.yaml
    severity_modifiers: dict        # Loaded from severity_modifiers.yaml
    entity_aliases: dict            # Loaded from entity_aliases.yaml
```

All dataclasses are immutable after construction.

### Path Resolution

`PROJECT_ROOT` is computed as two parent directories above `config.py` (i.e., the repository root). All config paths default to relative paths like `../cc-data/raw`, which are resolved against `PROJECT_ROOT` at runtime. Absolute paths provided via CLI flags are used as-is.

### Keyword Loading

The `_load_keyword_dicts` function loads all three YAML files from `config/keyword_dicts/`. Missing files are tolerated (empty dicts are used), allowing partial configurations during development.

---

## 10. Output

**Module:** `src/analysis/output.py`

### briefing.json Envelope

The `assemble_briefing` function constructs the top-level briefing dict with the following structure:

```json
{
  "date": "2025-01-30",
  "volume": 42,
  "signals": [
    {
      "id": "canada-china-ambassador-summoned",
      "title": {"en": "...", "zh": "..."},
      "body": {"en": "...", "zh": "..."},
      "source": {"en": "...", "zh": "..."},
      "date": "2025-01-30",
      "category": "diplomatic",
      "severity": "high"
    }
  ],
  "tension_index": {
    "composite": 6.2,
    "level": {"en": "Elevated", "zh": "..."},
    "delta": 0.3,
    "delta_description": {"en": "+0.3 from previous day", "zh": "..."},
    "components": [
      {
        "name": {"en": "Diplomatic", "zh": "..."},
        "score": 7,
        "weight": 0.25,
        "trend": "up",
        "key_driver": {"en": "Ambassador summoned", "zh": "..."}
      }
    ]
  },
  "trade_data": {"summary_stats": [], "commodities": []},
  "market_data": {"indices": [], "sectors": [], "movers": {...}, "ipos": []},
  "parliament": {"bills": [], "hansard": {...}},
  "entities": [...],
  "active_situations": [...],
  "quote_of_the_day": {"text": {...}, "attribution": {...}},
  "todays_number": {"value": {...}, "description": {...}},
  "disruptions": []
}
```

Optional fields (`pathway_cards`, `explore_cards`) are included only when explicitly provided. Supplementary sections (`trade_data`, `market_data`, `parliament`, `quote_of_the_day`, `todays_number`) fall back to minimal default structures when no upstream data is available.

### Schema Validation

Validation uses the `jsonschema` library with a `RefResolver` anchored to the schemas directory URI, enabling `$ref` resolution across multiple schema files. The validation behavior depends on configuration:

- If `validation.strict` is `true` (default in all environments) and validation fails, the CLI raises a `ClickException` and aborts.
- If the schema file is not found, validation is skipped with a warning.
- If `jsonschema` is not installed, validation is skipped gracefully.

### File Output

Two copies of `briefing.json` are written per run:

1. **Processed**: `{processed_dir}/{date}/briefing.json` -- the primary output for downstream consumers. A copy is also written to `{processed_dir}/latest/briefing.json` for convenient access.
2. **Archive**: `{archive_dir}/daily/{date}/briefing.json` -- the historical record used for trend computation and volume compilation.

All JSON output uses `ensure_ascii=False` to preserve Chinese characters and 2-space indentation for readability.

### Volume Number

The volume number is auto-incremented by scanning existing `archive/daily/*/briefing.json` files for the highest existing volume number.

---

## 11. Extension Points

### Adding a New Category

1. Add the category name and bilingual keyword lists to `config/keyword_dicts/categories.yaml`.
2. Add the category string to `VALID_CATEGORIES` in `src/analysis/classifiers/category.py`.
3. Insert it at the appropriate position in `SPECIFICITY_ORDER` for tiebreaking.
4. If the category should contribute to the tension index:
   - Add it to `COMPONENT_WEIGHTS` in `src/analysis/tension_index.py` and re-balance so weights sum to 1.0.
   - Add a bilingual name to `COMPONENT_NAMES_BILINGUAL`.
5. Add test cases to `tests/test_category.py`.

### Adjusting Tension Index Weights

1. Edit `COMPONENT_WEIGHTS` in `src/analysis/tension_index.py`.
2. Ensure all weights sum to exactly 1.0 (verified by `test_component_weights_sum_to_one`).
3. Adjust `cap_denominator` in the environment YAML config files to tune how quickly components saturate.

### Adding New Situation Triggers

1. Add a new entry to `KNOWN_SITUATIONS` in `src/analysis/active_situations.py`:

```python
{
    "id": "new_situation_id",
    "name": {"en": "New Situation Name", "zh": "(Chinese name)"},
    "trigger_keywords": ["keyword1", "keyword2", "(Chinese keyword)"],
    "default_severity": "elevated",
    "start_date": "2025-01-01",
}
```

2. No other code changes are needed -- the tracker automatically scans all signals against all situation trigger keywords.
3. Add test cases to `tests/test_active_situations.py`.

### Adding New Entities

1. Add an entry to `config/keyword_dicts/entity_aliases.yaml` with the canonical ID and bilingual aliases.
2. Optionally map the entity ID to a type in the `entity_types` dict in `src/analysis/entities.py`.
3. Add test cases to `tests/test_entities.py`.

### Adding New Source Tiers or Sources

1. Add source names to the appropriate tier in `SOURCE_TIERS` in `src/analysis/classifiers/source_mapper.py`.
2. The flat lookup table `_SOURCE_LOOKUP` is rebuilt automatically at import time.
3. To add a new tier entirely, also update `SOURCE_TIER_SCORES` in `src/analysis/classifiers/severity.py`.

### Adjusting Severity Thresholds

1. Edit `SEVERITY_THRESHOLDS` in `src/analysis/classifiers/severity.py` to change score-to-level boundaries.
2. Edit `SOURCE_TIER_SCORES` to change how much each source tier contributes.
3. Edit keyword weights in `config/keyword_dicts/severity_modifiers.yaml` to change escalation/de-escalation impact.

---

## 12. Testing

### Approach

The test suite uses **pytest** (v8.0+) with shared fixtures defined in `tests/conftest.py`. Tests load the actual keyword dictionaries from `config/keyword_dicts/`, ensuring that tests exercise the real data used in production. Temporary directories (`tmp_path`) are used for file I/O tests.

### Test Modules

| Module | What It Tests |
|---|---|
| `test_category.py` | Category classification for all 8 categories in both EN and ZH; bilingual signal classification; edge cases (empty text, string-only titles, headline-only signals, empty signals). |
| `test_severity.py` | Score-to-severity threshold mapping at all boundaries; individual factor contributions (source tier scoring, escalation keywords, bilateral directness, recency); de-escalation score reduction; score floor at zero (never negative); signal-level severity classification end-to-end. |
| `test_source_mapper.py` | Source name to tier mapping for known and unknown sources. |
| `test_tension_index.py` | Zero-signal baseline (composite=0.0, level=Low); single-component scoring; seed-data-equivalent composite verification (6.0--6.5, Elevated); maximum saturation (composite=10.0, Critical); delta computation from previous composite; component trend direction (up/down/stable); severity points constants; weight sum invariant (1.0); level threshold boundaries (Low, Moderate, Elevated); serialization via `to_dict()`. |
| `test_entities.py` | English and Chinese alias matching; specific entity matching (canola, Huawei, rare earths, Two Michaels); multi-entity signals with 7+ entities; implications field scanning; cross-signal aggregation with correct mention counting; sort order by mention count; entity directory structure (type, name, has_detail_page). |
| `test_active_situations.py` | Trigger detection for canola, tech decoupling, and foreign interference situations; no-match signals producing empty results; day count computation from historical start dates; severity upgrade from matching signals (elevated -> critical); output sorting by severity; single signal triggering multiple situations; serialization with and without optional fields. |
| `test_trend.py` | Day-over-day trend computation from previous briefings. |
| `test_volume_compiler.py` | Monthly volume compilation and aggregation. |
| `test_cli.py` | End-to-end `run` command with sample raw data (verifies briefing.json output structure); empty and nonexistent raw directories (graceful handling); `compile-volume` with and without archive data; help text and version output for all commands. |

### Shared Fixtures

`tests/conftest.py` provides:

- **Path fixtures**: `project_root`, `config_dir`, `keyword_dicts_dir`.
- **Dictionary fixtures**: `categories_dict`, `severity_modifiers`, `entity_aliases` -- loaded from actual YAML config files.
- **Signal fixtures**: six domain-specific sample signals (`sample_diplomatic_signal`, `sample_trade_signal`, `sample_military_signal`, `sample_tech_signal`, `sample_political_signal`, `sample_social_signal`), each with full bilingual title/body/source/date/implications structure.
- **Combined fixture**: `all_sample_signals` containing all six signals.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/test_tension_index.py

# Run a specific test class
pytest tests/test_category.py::TestClassifySignal

# Run a specific test
pytest tests/test_active_situations.py::TestTrackSituations::test_severity_upgrade
```

### Linting

The project uses **ruff** (v0.5+) for linting with the following configuration (from `pyproject.toml`):

- **Line length**: 100 characters.
- **Target**: Python 3.12.
- **Enabled rule sets**: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort import ordering), `N` (pep8-naming), `W` (pycodestyle warnings), `UP` (pyupgrade modernization).

```bash
ruff check src/ tests/
```

---

## Appendix: Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | >= 3.12 | Runtime |
| PyYAML | ^6.0 | YAML config and keyword dictionary loading |
| Click | ^8.1 | CLI framework |
| jsonschema | ^4.20 | briefing.json schema validation |
| pytest | ^8.0 | Test framework (dev dependency) |
| ruff | ^0.5 | Linter (dev dependency) |

**Build system:** Poetry (`poetry-core >= 1.0.0`).

**Entry point:** `analysis = "analysis.cli:main"` (registered via `[tool.poetry.scripts]`).
