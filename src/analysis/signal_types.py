"""Structured type definitions for signals at each pipeline stage.

TypedDict preserves dict compatibility — no existing code rewrites needed.
These types document the expected shape of signal dicts as they flow
through the pipeline: Raw → Classified → Normalized.
"""

from __future__ import annotations

from typing import Any, TypedDict


class BilingualText(TypedDict):
    """A text field with English and Chinese variants."""

    en: str
    zh: str


class RawSignal(TypedDict, total=False):
    """Signal as loaded from fetcher JSON files.

    Fields vary by source; all are optional (total=False).
    """

    title: str | BilingualText
    body: str | BilingualText
    body_text: str
    body_snippet: str
    source: str | BilingualText
    date: str
    url: str
    source_url: str
    language: str  # "en" or "zh"
    region: str  # "mainland", "taiwan", "hongkong"
    id: str


class ClassifiedSignal(TypedDict, total=False):
    """Signal after category/severity classification and entity matching."""

    title: str | BilingualText
    body: str | BilingualText
    source: str | BilingualText
    date: str
    url: str
    id: str
    category: str
    severity: str
    entity_ids: list[str]
    source_tier: str
    language: str
    region: str


class NormalizedSignal(TypedDict, total=False):
    """Signal after normalization to bilingual schema format.

    All text fields are BilingualText; implications and perspectives
    are fully populated.
    """

    id: str
    title: BilingualText
    body: BilingualText
    source: BilingualText
    date: str
    category: str
    severity: str
    entity_ids: list[str]
    implications: dict[str, Any]
    perspectives: dict[str, Any]
    original_zh_source: bool
    original_zh_url: str
