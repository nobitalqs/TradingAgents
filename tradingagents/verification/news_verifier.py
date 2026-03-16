"""News credibility assessment via source tiers and suspicious-pattern detection."""

from __future__ import annotations

import re

from tradingagents.verification.models import NewsCredibility

# Regex flags used for all pattern matching.
_RE_FLAGS = re.IGNORECASE


class NewsVerifier:
    """Assess credibility of financial news items."""

    T1_SOURCES: frozenset[str] = frozenset(
        {
            "reuters",
            "bloomberg",
            "wsj",
            "wall street journal",
            "ft",
            "financial times",
            "sec.gov",
            "federalreserve.gov",
            "associated press",
            "ap news",
        }
    )

    T2_SOURCES: frozenset[str] = frozenset(
        {
            "cnbc",
            "marketwatch",
            "yahoo finance",
            "barrons",
            "seeking alpha",
            "benzinga",
        }
    )

    SUSPICIOUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"guaranteed\s+profit", _RE_FLAGS), "guaranteed_profit"),
        (re.compile(r"insider\s+exclusive", _RE_FLAGS), "insider_exclusive"),
        (re.compile(r"breaking", _RE_FLAGS), "exaggerated_breaking"),
        (re.compile(r"secret\s+tip", _RE_FLAGS), "secret_tip"),
        (
            re.compile(r"act\s+now|limited\s+time|don'?t\s+miss", _RE_FLAGS),
            "urgency_pressure",
        ),
        (re.compile(r"anonymous\s+source", _RE_FLAGS), "anonymous_source"),
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        headline: str,
        source: str,
        body: str = "",
    ) -> NewsCredibility:
        """Assess the credibility of a single news item.

        Parameters
        ----------
        headline:
            The news headline text.
        source:
            Publisher / source name.
        body:
            Optional article body text to scan for suspicious patterns.

        Returns
        -------
        NewsCredibility with a score clamped to [0, 1].
        """
        source_lower = source.strip().lower()
        text_to_scan = f"{headline} {body}"

        # --- Determine tier and baseline adjustment ---
        tier, tier_delta = self._classify_source(source_lower)

        # --- Detect suspicious patterns ---
        flags: list[str] = []
        for pattern, label in self.SUSPICIOUS_PATTERNS:
            if pattern.search(text_to_scan):
                flags.append(label)

        # --- Compute score ---
        score = 0.7 + tier_delta - 0.15 * len(flags)
        score = max(0.0, min(1.0, score))

        return NewsCredibility(
            score=score,
            flags=flags,
            source_tier=tier,
            headline=headline,
            source_name=source,
        )

    def assess_batch(
        self,
        items: list[tuple[str, str]],
    ) -> list[NewsCredibility]:
        """Assess a batch of (headline, source) pairs."""
        return [self.assess(headline, source) for headline, source in items]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_source(self, source_lower: str) -> tuple[str, float]:
        """Return (tier_label, score_delta) for the given source name."""
        if source_lower in self.T1_SOURCES:
            return "T1", 0.2
        if source_lower in self.T2_SOURCES:
            return "T2", 0.1
        return "unknown", -0.3
