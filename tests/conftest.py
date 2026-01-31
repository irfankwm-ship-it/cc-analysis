"""Shared fixtures for cc-analysis tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory."""
    return project_root / "config"


@pytest.fixture
def keyword_dicts_dir(config_dir: Path) -> Path:
    """Return the keyword_dicts directory."""
    return config_dir / "keyword_dicts"


@pytest.fixture
def categories_dict(
    keyword_dicts_dir: Path,
) -> dict[str, dict[str, list[str]]]:
    """Load categories.yaml."""
    path = keyword_dicts_dir / "categories.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def severity_modifiers(keyword_dicts_dir: Path) -> dict[str, Any]:
    """Load severity_modifiers.yaml."""
    path = keyword_dicts_dir / "severity_modifiers.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def entity_aliases(
    keyword_dicts_dir: Path,
) -> dict[str, dict[str, list[str]]]:
    """Load entity_aliases.yaml."""
    path = keyword_dicts_dir / "entity_aliases.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_diplomatic_signal() -> dict[str, Any]:
    """A sample diplomatic signal matching seed data patterns."""
    return {
        "id": "canada-china-ambassador-summoned",
        "title": {
            "en": (
                "Canada Summons Chinese Ambassador"
                " Over Diplomatic Dispute"
            ),
            "zh": (
                "\u52A0\u62FF\u5927\u5C31\u5916\u4EA4\u4E89\u7AEF"
                "\u53EC\u89C1\u4E2D\u56FD\u5927\u4F7F"
            ),
        },
        "body": {
            "en": (
                "Global Affairs Canada summoned the Chinese"
                " ambassador to discuss bilateral tensions"
                " following diplomatic incidents. The MFA"
                " spokesperson responded with strong opposition."
            ),
            "zh": (
                "\u52A0\u62FF\u5927\u5168\u7403\u4E8B\u52A1\u90E8"
                "\u53EC\u89C1\u4E2D\u56FD\u5927\u4F7F\uFF0C"
                "\u8BA8\u8BBA\u5916\u4EA4\u4E8B\u4EF6\u5F15\u53D1"
                "\u7684\u53CC\u8FB9\u7D27\u5F20\u5C40\u52BF\u3002"
                "\u5916\u4EA4\u90E8\u53D1\u8A00\u4EBA"
                "\u8868\u793A\u5F3A\u70C8\u53CD\u5BF9\u3002"
            ),
        },
        "source": {
            "en": "Global Affairs Canada",
            "zh": "\u52A0\u62FF\u5927\u5168\u7403\u4E8B\u52A1\u90E8",
        },
        "date": "2025-01-30",
        "implications": {
            "canada_impact": {
                "en": (
                    "Direct impact on Canada-China"
                    " diplomatic relations."
                ),
                "zh": (
                    "\u76F4\u63A5\u5F71\u54CD\u52A0\u4E2D"
                    "\u5916\u4EA4\u5173\u7CFB\u3002"
                ),
            },
        },
    }


@pytest.fixture
def sample_trade_signal() -> dict[str, Any]:
    """A sample trade signal."""
    return {
        "id": "canola-tariff-increase",
        "title": {
            "en": (
                "China Imposes New Tariff on"
                " Canadian Canola Imports"
            ),
            "zh": (
                "\u4E2D\u56FD\u5BF9\u52A0\u62FF\u5927\u6CB9\u83DC"
                "\u7C7D\u8FDB\u53E3\u52A0\u5F81\u65B0\u5173\u7A0E"
            ),
        },
        "body": {
            "en": (
                "MOFCOM announced a 25% tariff increase on"
                " Canadian canola seed imports, citing trade"
                " concerns. This sanctions move escalates"
                " the ongoing trade war between the two"
                " nations."
            ),
            "zh": (
                "\u5546\u52A1\u90E8\u5BA3\u5E03\u5BF9\u52A0\u62FF"
                "\u5927\u6CB9\u83DC\u7C7D\u8FDB\u53E3\u52A0\u5F81"
                "25%\u5173\u7A0E\uFF0C\u79F0\u51FA\u4E8E\u8D38"
                "\u6613\u5173\u5207\u3002\u8FD9\u4E00\u5236\u88C1"
                "\u4E3E\u63AA\u5347\u7EA7\u4E86\u4E24\u56FD\u4E4B"
                "\u95F4\u7684\u8D38\u6613\u6218\u3002"
            ),
        },
        "source": {
            "en": "MOFCOM",
            "zh": "\u5546\u52A1\u90E8",
        },
        "date": "2025-01-30",
        "implications": {
            "canada_impact": {
                "en": (
                    "Significant impact on Canadian"
                    " agricultural exports."
                ),
                "zh": (
                    "\u5BF9\u52A0\u62FF\u5927\u519C\u4EA7\u54C1"
                    "\u51FA\u53E3\u4EA7\u751F\u91CD\u5927"
                    "\u5F71\u54CD\u3002"
                ),
            },
        },
    }


@pytest.fixture
def sample_military_signal() -> dict[str, Any]:
    """A sample military signal."""
    return {
        "id": "pla-taiwan-strait-exercise",
        "title": {
            "en": (
                "PLA Conducts Military Exercise"
                " in Taiwan Strait"
            ),
            "zh": (
                "\u89E3\u653E\u519B\u5728\u53F0\u6E7E\u6D77\u5CE1"
                "\u8FDB\u884C\u519B\u4E8B\u6F14\u4E60"
            ),
        },
        "body": {
            "en": (
                "The PLA navy deployed warships for a major"
                " military exercise in the Taiwan Strait."
                " NORAD tracked increased air force activity"
                " in the region."
            ),
            "zh": (
                "\u89E3\u653E\u519B\u6D77\u519B\u90E8\u7F72"
                "\u519B\u8230\u5728\u53F0\u6E7E\u6D77\u5CE1"
                "\u8FDB\u884C\u5927\u89C4\u6A21\u519B\u4E8B"
                "\u6F14\u4E60\u3002\u5317\u7F8E\u9632\u7A7A"
                "\u8FFD\u8E2A\u5230\u8BE5\u5730\u533A\u7A7A"
                "\u519B\u6D3B\u52A8\u589E\u52A0\u3002"
            ),
        },
        "source": {
            "en": "Reuters",
            "zh": "\u8DEF\u900F\u793E",
        },
        "date": "2025-01-30",
        "implications": {
            "canada_impact": {
                "en": (
                    "Regional security implications"
                    " for Canadian interests."
                ),
                "zh": (
                    "\u5BF9\u52A0\u62FF\u5927\u5229\u76CA"
                    "\u4EA7\u751F\u5730\u533A\u5B89\u5168"
                    "\u5F71\u54CD\u3002"
                ),
            },
        },
    }


@pytest.fixture
def sample_tech_signal() -> dict[str, Any]:
    """A sample technology signal."""
    return {
        "id": "rare-earth-export-restriction",
        "title": {
            "en": (
                "China Restricts Rare Earth Exports"
                " Including Gallium and Germanium"
            ),
            "zh": (
                "\u4E2D\u56FD\u9650\u5236\u7A00\u571F\u51FA\u53E3"
                "\uFF0C\u5305\u62EC\u9553\u548C\u9574"
            ),
        },
        "body": {
            "en": (
                "China's MOFCOM announced new export"
                " restrictions on rare earth elements"
                " including gallium and germanium,"
                " tightening the semiconductor supply chain."
            ),
            "zh": (
                "\u4E2D\u56FD\u5546\u52A1\u90E8\u5BA3\u5E03"
                "\u5BF9\u7A00\u571F\u5143\u7D20\u5B9E\u65BD"
                "\u65B0\u7684\u51FA\u53E3\u9650\u5236\uFF0C"
                "\u5305\u62EC\u9553\u548C\u9574\uFF0C\u6536\u7D27"
                "\u534A\u5BFC\u4F53\u4F9B\u5E94\u94FE\u3002"
            ),
        },
        "source": {
            "en": "Bloomberg",
            "zh": "\u5F6D\u535A",
        },
        "date": "2025-01-30",
        "implications": {
            "canada_impact": {
                "en": (
                    "Canada's critical minerals strategy"
                    " gains importance."
                ),
                "zh": (
                    "\u52A0\u62FF\u5927\u5173\u952E\u77FF\u4EA7"
                    "\u6218\u7565\u66F4\u52A0\u91CD\u8981\u3002"
                ),
            },
        },
    }


@pytest.fixture
def sample_political_signal() -> dict[str, Any]:
    """A sample political signal."""
    return {
        "id": "foreign-interference-inquiry",
        "title": {
            "en": (
                "Foreign Interference Inquiry Examines"
                " CCP Influence on Canadian Elections"
            ),
            "zh": (
                "\u5916\u56FD\u5E72\u9884\u8C03\u67E5\u5BA1\u67E5"
                "\u4E2D\u5171\u5BF9\u52A0\u62FF\u5927\u9009\u4E3E"
                "\u7684\u5F71\u54CD"
            ),
        },
        "body": {
            "en": (
                "Parliament's foreign interference inquiry"
                " heard testimony about CCP influence"
                " operations targeting Canadian elections."
                " CSIS provided classified evidence to the"
                " committee."
            ),
            "zh": (
                "\u8BAE\u4F1A\u5916\u56FD\u5E72\u9884\u8C03\u67E5"
                "\u542C\u53D6\u4E86\u5173\u4E8E\u4E2D\u5171"
                "\u5F71\u54CD\u52A0\u62FF\u5927\u9009\u4E3E"
                "\u7684\u8BC1\u8BCD\u3002\u52A0\u62FF\u5927"
                "\u5B89\u5168\u60C5\u62A5\u5C40\u5411\u59D4"
                "\u5458\u4F1A\u63D0\u4F9B\u4E86\u673A\u5BC6"
                "\u8BC1\u636E\u3002"
            ),
        },
        "source": {"en": "CBC", "zh": "CBC"},
        "date": "2025-01-30",
        "implications": {
            "canada_impact": {
                "en": (
                    "Critical for Canadian"
                    " democratic integrity."
                ),
                "zh": (
                    "\u5BF9\u52A0\u62FF\u5927\u6C11\u4E3B"
                    "\u5B8C\u6574\u6027\u81F3\u5173\u91CD"
                    "\u8981\u3002"
                ),
            },
        },
    }


@pytest.fixture
def sample_social_signal() -> dict[str, Any]:
    """A sample social signal."""
    return {
        "id": "confucius-institute-review",
        "title": {
            "en": (
                "University Reviews Confucius"
                " Institute Partnership"
            ),
            "zh": (
                "\u5927\u5B66\u5BA1\u67E5\u5B54\u5B50"
                "\u5B66\u9662\u5408\u4F5C"
            ),
        },
        "body": {
            "en": (
                "A major Canadian university announced"
                " a review of its Confucius Institute"
                " partnership amid surveillance concerns"
                " and diaspora community pressure."
            ),
            "zh": (
                "\u4E00\u6240\u52A0\u62FF\u5927\u4E3B\u8981"
                "\u5927\u5B66\u5BA3\u5E03\u5BA1\u67E5\u5176"
                "\u5B54\u5B50\u5B66\u9662\u5408\u4F5C\uFF0C"
                "\u56E0\u76D1\u63A7\u62C5\u5FE7\u548C\u4FA8"
                "\u6C11\u793E\u533A\u538B\u529B\u3002"
            ),
        },
        "source": {
            "en": "Globe and Mail",
            "zh": "\u73AF\u7403\u90AE\u62A5",
        },
        "date": "2025-01-30",
        "implications": {
            "canada_impact": {
                "en": (
                    "Impacts Chinese student community"
                    " and academic relations."
                ),
                "zh": (
                    "\u5F71\u54CD\u4E2D\u56FD\u7559\u5B66\u751F"
                    "\u7FA4\u4F53\u548C\u5B66\u672F\u5173"
                    "\u7CFB\u3002"
                ),
            },
        },
    }


@pytest.fixture
def all_sample_signals(
    sample_diplomatic_signal: dict[str, Any],
    sample_trade_signal: dict[str, Any],
    sample_military_signal: dict[str, Any],
    sample_tech_signal: dict[str, Any],
    sample_political_signal: dict[str, Any],
    sample_social_signal: dict[str, Any],
) -> list[dict[str, Any]]:
    """All sample signals combined."""
    return [
        sample_diplomatic_signal,
        sample_trade_signal,
        sample_military_signal,
        sample_tech_signal,
        sample_political_signal,
        sample_social_signal,
    ]
