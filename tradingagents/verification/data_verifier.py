"""High-level data verification combining price consistency and news credibility."""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.verification.models import (
    CredibilitySummary,
    NewsCredibility,
    VerifiedDataPoint,
)
from tradingagents.verification.news_verifier import NewsVerifier

logger = logging.getLogger(__name__)


class DataVerifier:
    """Orchestrates multi-source data verification."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._news_verifier = NewsVerifier()

    # ------------------------------------------------------------------
    # Price consistency
    # ------------------------------------------------------------------

    def _assess_price_consistency(
        self,
        prices: dict[str, float | None],
    ) -> VerifiedDataPoint:
        """Assess consistency of price data across sources.

        Rules
        -----
        - No valid prices → confidence 0.0
        - Single valid source → confidence 0.5
        - Multiple sources, max deviation ≤ 1% → 0.95
        - Multiple sources, max deviation ≤ 2% → 0.7
        - Otherwise → 0.5
        """
        # Filter out None values.
        valid: dict[str, float] = {k: v for k, v in prices.items() if v is not None}

        if not valid:
            return VerifiedDataPoint(
                value=None,
                sources=[],
                confidence=0.0,
                discrepancies=["No valid price data available"],
            )

        sources = list(valid.keys())
        values = list(valid.values())
        mean_price = sum(values) / len(values)

        if len(valid) == 1:
            return VerifiedDataPoint(
                value=values[0],
                sources=sources,
                confidence=0.5,
                discrepancies=[],
            )

        # Compute max deviation from the mean.
        max_deviation = max(abs(v - mean_price) / mean_price for v in values)
        deviation_pct = max_deviation * 100

        discrepancies: list[str] = []
        if deviation_pct <= 1.0:
            confidence = 0.95
        elif deviation_pct <= 2.0:
            confidence = 0.7
        else:
            confidence = 0.5
            discrepancies.append(
                f"Price deviation of {deviation_pct:.2f}% across sources"
            )

        return VerifiedDataPoint(
            value=mean_price,
            sources=sources,
            confidence=confidence,
            discrepancies=discrepancies,
        )

    # ------------------------------------------------------------------
    # News assessment
    # ------------------------------------------------------------------

    def assess_news(
        self,
        news_items: list[dict[str, str]],
    ) -> list[NewsCredibility]:
        """Assess a list of news item dicts (with 'headline' and 'source' keys)."""
        if not news_items:
            return []

        pairs: list[tuple[str, str]] = [
            (item.get("headline", ""), item.get("source", "")) for item in news_items
        ]
        return self._news_verifier.assess_batch(pairs)

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def build_credibility_summary(
        self,
        price_data: dict[str, float | None] | None = None,
        news_items: list[dict[str, str]] | None = None,
    ) -> CredibilitySummary:
        """Build a complete credibility summary for an analysis cycle."""
        # --- Price confidence ---
        if price_data is not None:
            price_point = self._assess_price_consistency(price_data)
            price_confidence = price_point.confidence
        else:
            price_confidence = 1.0

        # --- News credibility ---
        assessed_news: list[NewsCredibility] = []
        if news_items is not None:
            assessed_news = self.assess_news(news_items)

        # --- Warnings ---
        warnings: list[str] = []
        unreliable = [n for n in assessed_news if not n.is_reliable]
        if unreliable:
            warnings.append(
                f"{len(unreliable)} of {len(assessed_news)} news items "
                f"have low credibility (score < 0.5)"
            )

        if price_data is not None and price_confidence < 0.7:
            warnings.append(f"Price data has low confidence ({price_confidence:.2f})")

        return CredibilitySummary(
            price_confidence=price_confidence,
            news_items=assessed_news,
            warnings=warnings,
        )
