"""Tests for data_transforms module."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.data_transforms import (
    determine_volume_number,
    extract_market_signals,
    generate_quote,
    generate_todays_number,
    is_regulatory,
    load_supplementary_data,
    transform_market_data,
    transform_parliament_data,
    transform_trade_data,
)


class TestTransformMarketData:
    def test_basic_transform(self) -> None:
        raw = {
            "indices": [{"name": "HSI", "value": 20000, "change_pct": 1.5}],
            "sectors": [],
            "movers": {"gainers": [], "losers": []},
            "currency_pairs": [],
        }
        result = transform_market_data(raw)
        assert len(result["indices"]) == 1
        assert result["indices"][0]["direction"] == "up"
        assert "+1.50%" in result["indices"][0]["change"]

    def test_sparkline_conversion(self) -> None:
        raw = {
            "indices": [
                {"name": "HSI", "value": 20000, "change_pct": 0,
                 "sparkline": [100, 105, 102, 110]},
            ],
            "sectors": [], "movers": {"gainers": [], "losers": []},
            "currency_pairs": [],
        }
        result = transform_market_data(raw)
        assert result["indices"][0]["sparkline_points"] != ""

    def test_currency_pairs(self) -> None:
        raw = {
            "indices": [], "sectors": [],
            "movers": {"gainers": [], "losers": []},
            "currency_pairs": [
                {"name": "USD/CNY", "rate": 7.2345, "change_pct": 0.01},
            ],
        }
        result = transform_market_data(raw)
        assert len(result["currency_pairs"]) == 1
        assert "7.2345" in result["currency_pairs"][0]["rate"]


class TestTransformTradeData:
    def test_basic_transform(self) -> None:
        raw = {
            "imports_cad_millions": 5000,
            "exports_cad_millions": 3000,
            "balance_cad_millions": -2000,
            "commodities": [],
        }
        result = transform_trade_data(raw)
        assert len(result["summary_stats"]) == 3
        assert result["summary_stats"][2]["direction"] == "down"

    def test_commodity_table(self) -> None:
        raw = {
            "imports_cad_millions": 0,
            "exports_cad_millions": 0,
            "balance_cad_millions": 0,
            "commodities": [
                {"name": "Canola", "export_cad_millions": 500,
                 "import_cad_millions": 10, "trend": "up"},
            ],
        }
        result = transform_trade_data(raw)
        assert len(result["commodity_table"]) == 1
        assert result["commodity_table"][0]["trend"]["en"] == "Increasing"


class TestTransformParliamentData:
    def test_basic_transform(self) -> None:
        raw = {
            "bills": [{"title": "Bill C-70", "status": "HouseInCommittee"}],
            "hansard_stats": {"total_mentions": 5, "by_keyword": {"china": 3, "trade": 2}},
        }
        result = transform_parliament_data(raw)
        assert len(result["bills"]) == 1
        assert result["bills"][0]["status"]["en"] == "In Committee"
        assert result["hansard"]["session_mentions"] == 5
        assert result["hansard"]["top_topic"]["en"] == "china"


class TestLoadSupplementaryData:
    def test_loads_market_data(self, tmp_path: Path) -> None:
        market = {
            "data": {
                "indices": [{"name": "HSI", "value": 20000, "change_pct": 1.5}],
                "sectors": [], "movers": {"gainers": [], "losers": []},
                "currency_pairs": [],
            }
        }
        with open(tmp_path / "yahoo_finance.json", "w") as f:
            json.dump(market, f)
        result = load_supplementary_data(str(tmp_path))
        assert result["market_data"] is not None

    def test_handles_missing_files(self, tmp_path: Path) -> None:
        result = load_supplementary_data(str(tmp_path))
        assert result["market_data"] is None
        assert result["trade_data"] is None


class TestDetermineVolumeNumber:
    def test_empty_archive(self, tmp_path: Path) -> None:
        assert determine_volume_number(str(tmp_path)) == 1

    def test_increments_volume(self, tmp_path: Path) -> None:
        daily = tmp_path / "daily" / "2025-01-30"
        daily.mkdir(parents=True)
        with open(daily / "briefing.json", "w") as f:
            json.dump({"volume": 5}, f)
        assert determine_volume_number(str(tmp_path)) == 6


class TestGenerateTodaysNumber:
    def test_from_trade_data(self) -> None:
        supplementary = {
            "trade_data": {
                "totals": {"total_imports_cad": 5000, "total_exports_cad": 3000},
                "reference_period": "2025-01-01",
            }
        }
        result = generate_todays_number(supplementary, [])
        assert "B" in result["value"]["en"] or "M" in result["value"]["en"]

    def test_fallback_to_signal_count(self) -> None:
        result = generate_todays_number({}, [{"id": 1}, {"id": 2}])
        assert result["value"]["en"] == "2"


class TestIsRegulatory:
    def test_regulatory_signal(self) -> None:
        assert is_regulatory({"title": "SAMR launches antitrust investigation"})

    def test_non_regulatory(self) -> None:
        assert not is_regulatory({"title": "China announces new policy"})


class TestExtractMarketSignals:
    def test_extracts_by_category(self) -> None:
        signals = [
            {"category": "trade", "severity": "high", "title": "Trade news"},
            {"category": "diplomatic", "severity": "high", "title": "Diplomatic news"},
            {"category": "economic", "severity": "moderate", "title": "Economic news"},
        ]
        market, regulatory = extract_market_signals(signals)
        assert len(market) == 2  # trade + economic


class TestGenerateQuote:
    def test_prefers_bilateral(self) -> None:
        signals = [
            {"title": {"en": "Canada and China sign deal", "zh": "加中签署协议"},
             "source": {"en": "Reuters", "zh": "路透"}, "severity": "high", "date": "2025-01-30"},
            {"title": {"en": "Weather report", "zh": "天气"},
             "source": {"en": "Local", "zh": "本地"}, "severity": "low", "date": "2025-01-30"},
        ]
        result = generate_quote(signals)
        assert "Canada" in result["text"]["en"]

    def test_empty_signals(self) -> None:
        result = generate_quote([])
        assert result["text"]["en"] == ""
