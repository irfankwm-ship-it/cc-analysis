"""Tension index computation.

Implements the Canada-China Tension Index formula:
  - For each component, sum severity points of all signals within 30-day window
  - component_score = min(sum / 20 * 10, 10), capped at 10
  - composite = sum(component_score * weight) for all 6 tracked components
  - Level mapping based on composite score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

COMPONENT_WEIGHTS: dict[str, float] = {
    "diplomatic": 0.25,
    "trade": 0.25,
    "military": 0.15,
    "political": 0.15,
    "technology": 0.10,
    "social": 0.10,
}

SEVERITY_POINTS: dict[str, int] = {
    "critical": 5,
    "high": 4,
    "elevated": 3,
    "moderate": 2,
    "low": 1,
}

COMPONENT_NAMES_BILINGUAL: dict[str, dict[str, str]] = {
    "diplomatic": {"en": "Diplomatic", "zh": "\u5916\u4EA4"},
    "trade": {"en": "Trade", "zh": "\u8D38\u6613"},
    "military": {"en": "Military", "zh": "\u519B\u4E8B"},
    "political": {"en": "Political", "zh": "\u653F\u6CBB"},
    "technology": {"en": "Technology", "zh": "\u79D1\u6280"},
    "social": {"en": "Social", "zh": "\u793E\u4F1A"},
}

LEVEL_THRESHOLDS: list[tuple[float, str, str]] = [
    (9.0, "Critical", "\u5371\u6025"),
    (7.0, "High", "\u9AD8"),
    (4.1, "Elevated", "\u5347\u9AD8"),
    (2.1, "Moderate", "\u4E2D\u7B49"),
    (0.0, "Low", "\u4F4E"),
]


@dataclass
class ComponentScore:
    """A single component of the tension index."""

    name: dict[str, str]
    score: int
    weight: float
    trend: str = "stable"
    key_driver: dict[str, str] = field(default_factory=lambda: {"en": "", "zh": ""})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching schema."""
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "trend": self.trend,
            "key_driver": self.key_driver,
        }


@dataclass
class TensionIndex:
    """Complete tension index with composite and component breakdown."""

    composite: float
    level: dict[str, str]
    delta: float
    delta_description: dict[str, str]
    components: list[ComponentScore]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching schema."""
        return {
            "composite": self.composite,
            "level": self.level,
            "delta": self.delta,
            "delta_description": self.delta_description,
            "components": [c.to_dict() for c in self.components],
        }


def _compute_level(composite: float) -> dict[str, str]:
    """Map composite score to a bilingual level label."""
    for threshold, en_label, zh_label in LEVEL_THRESHOLDS:
        if composite >= threshold:
            return {"en": en_label, "zh": zh_label}
    return {"en": "Low", "zh": "\u4F4E"}


def _compute_delta_description(delta: float) -> dict[str, str]:
    """Generate bilingual description of the delta."""
    if delta > 0:
        return {
            "en": f"+{delta:.1f} from previous day",
            "zh": f"\u6BD4\u524D\u4E00\u5929+{delta:.1f}",
        }
    if delta < 0:
        return {
            "en": f"{delta:.1f} from previous day",
            "zh": f"\u6BD4\u524D\u4E00\u5929{delta:.1f}",
        }
    return {
        "en": "No change from previous day",
        "zh": "\u4E0E\u524D\u4E00\u5929\u6301\u5E73",
    }


def _find_key_driver(
    signals: list[dict[str, Any]],
    category: str,
) -> dict[str, str]:
    """Find the key driver signal for a component category.

    Returns the title of the highest-severity signal in this category.
    """
    category_signals = [s for s in signals if s.get("category") == category]
    if not category_signals:
        return {"en": "No significant activity", "zh": "\u65E0\u91CD\u5927\u6D3B\u52A8"}

    # Sort by severity (critical > high > elevated > moderate > low)
    severity_order = {"critical": 5, "high": 4, "elevated": 3, "moderate": 2, "low": 1}
    category_signals.sort(
        key=lambda s: severity_order.get(s.get("severity", "low"), 0),
        reverse=True,
    )

    top_signal = category_signals[0]
    title = top_signal.get("title", {"en": "", "zh": ""})
    if isinstance(title, str):
        return {"en": title, "zh": title}
    return {
        "en": title.get("en", ""),
        "zh": title.get("zh", ""),
    }


def compute_tension_index(
    signals: list[dict[str, Any]],
    previous_composite: float | None = None,
    previous_components: dict[str, int] | None = None,
    cap_denominator: int = 20,
) -> TensionIndex:
    """Compute the tension index from classified signals.

    Args:
        signals: List of signal dicts with 'category' and 'severity' fields.
        previous_composite: Previous day's composite score for delta calc.
        previous_components: Previous day's component scores for trend calc.
        cap_denominator: Denominator for score capping (default 20).

    Returns:
        TensionIndex object with composite, level, delta, and components.
    """
    # Sum severity points per component category
    category_points: dict[str, int] = {cat: 0 for cat in COMPONENT_WEIGHTS}

    for signal in signals:
        cat = signal.get("category", "")
        sev = signal.get("severity", "low")
        if cat in category_points:
            category_points[cat] += SEVERITY_POINTS.get(sev, 0)

    # Compute component scores
    components: list[ComponentScore] = []
    composite = 0.0

    for cat, weight in COMPONENT_WEIGHTS.items():
        raw_score = category_points[cat]
        # component_score = min(sum / cap_denominator * 10, 10)
        component_score = min(raw_score / cap_denominator * 10, 10)
        # Round to integer for the schema (score is integer type)
        int_score = min(round(component_score), 10)

        # Determine trend from previous day
        trend = "stable"
        if previous_components and cat in previous_components:
            prev_score = previous_components[cat]
            if int_score > prev_score:
                trend = "up"
            elif int_score < prev_score:
                trend = "down"

        key_driver = _find_key_driver(signals, cat)

        components.append(
            ComponentScore(
                name=COMPONENT_NAMES_BILINGUAL[cat],
                score=int_score,
                weight=weight,
                trend=trend,
                key_driver=key_driver,
            )
        )

        composite += component_score * weight

    # Round composite to 1 decimal place
    composite = round(composite, 1)

    # Compute delta
    delta = 0.0
    if previous_composite is not None:
        delta = round(composite - previous_composite, 1)

    level = _compute_level(composite)
    delta_description = _compute_delta_description(delta)

    return TensionIndex(
        composite=composite,
        level=level,
        delta=delta,
        delta_description=delta_description,
        components=components,
    )
