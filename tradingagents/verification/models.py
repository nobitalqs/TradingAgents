"""Frozen dataclasses for the verification layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerifiedDataPoint:
    """A data point that has been verified across multiple sources."""

    value: Any
    sources: list[str]
    confidence: float
    discrepancies: list[str]


@dataclass(frozen=True)
class NewsCredibility:
    """Credibility assessment for a single news item."""

    score: float
    flags: list[str]
    source_tier: str
    headline: str = ""
    source_name: str = ""

    @property
    def is_reliable(self) -> bool:
        return self.score >= 0.5


@dataclass(frozen=True)
class CredibilitySummary:
    """Aggregated credibility summary for an analysis cycle."""

    price_confidence: float = 1.0
    news_items: list[NewsCredibility] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format a DATA RELIABILITY ASSESSMENT block for LLM prompts."""
        lines: list[str] = []
        lines.append("=== DATA RELIABILITY ASSESSMENT ===")
        lines.append(f"Price confidence: {self.price_confidence}")
        lines.append("")

        if self.news_items:
            lines.append("News credibility:")
            for item in self.news_items:
                reliable_tag = "RELIABLE" if item.is_reliable else "UNRELIABLE"
                source_label = item.source_name or "unknown"
                headline_label = item.headline or "(no headline)"
                lines.append(
                    f"  [{reliable_tag}] {source_label} — "
                    f"{headline_label} (score: {item.score:.2f}, "
                    f"tier: {item.source_tier})"
                )
                if item.flags:
                    lines.append(f"    Flags: {', '.join(item.flags)}")
        else:
            lines.append("No news items assessed.")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        lines.append("=== END ASSESSMENT ===")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "price_confidence": self.price_confidence,
            "news_items": [
                {
                    "score": item.score,
                    "flags": item.flags,
                    "source_tier": item.source_tier,
                    "headline": item.headline,
                    "source_name": item.source_name,
                    "is_reliable": item.is_reliable,
                }
                for item in self.news_items
            ],
            "warnings": list(self.warnings),
        }
