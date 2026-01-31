# cc-analysis

Rule-based analysis engine for the China Compass pipeline. Processes raw fetched data and produces structured `briefing.json` output.

## Pipeline

```
Raw JSON (from cc-fetcher) -> classify -> score -> index -> trend -> entities -> situations -> briefing.json
```

## Usage

```bash
# Run full analysis for today
analysis run

# Run for a specific date and environment
analysis run --env staging --date 2025-01-30

# Compile monthly volume
analysis compile-volume --date 2025-01-31

# Custom directories
analysis run --raw-dir ./data/raw/2025-01-30 --output-dir ./data/processed/2025-01-30
```

## Configuration

Config files live in `config/analysis.{env}.yaml`. The environment is selected via:
1. `--env` CLI flag
2. `CC_ENV` environment variable
3. Defaults to `dev`

Keyword dictionaries are in `config/keyword_dicts/`.

## Development

```bash
poetry install
poetry run pytest
poetry run ruff check src/ tests/
```

## License

Data is provided for demonstration and development purposes.
