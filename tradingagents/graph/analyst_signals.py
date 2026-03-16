"""Analyst signal extraction and consensus computation.

Extracts directional signals from analyst reports and computes
a consensus view with confidence scoring.
"""

import logging
import re
from dataclasses import dataclass

from tradingagents.graph.signal_processing import extract_signal

logger = logging.getLogger("tradingagents.graph.analyst_signals")

_DIRECTION_RE = re.compile(r"DIRECTION\s*:\s*(BUY|SELL|HOLD)", re.IGNORECASE)

_VALID_DIRECTIONS = frozenset({"BUY", "SELL", "HOLD", "NEUTRAL"})

_ANALYST_REPORT_KEYS = {
    "market_analyst": "market_report",
    "sentiment_analyst": "sentiment_report",
    "news_analyst": "news_report",
    "fundamentals_analyst": "fundamentals_report",
}


@dataclass(frozen=True)
class AnalystSignal:
    """A single analyst's directional signal."""

    analyst: str
    direction: str  # "BUY" | "SELL" | "HOLD" | "NEUTRAL"
    key_reason: str


@dataclass(frozen=True)
class AnalystConsensus:
    """Aggregated consensus across multiple analyst signals."""

    signals: tuple[AnalystSignal, ...]
    buy_count: int
    sell_count: int
    hold_count: int
    neutral_count: int
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"

    @property
    def to_dict(self) -> dict:
        """Serialize consensus to a plain dictionary."""
        return {
            "signals": [
                {
                    "analyst": s.analyst,
                    "direction": s.direction,
                    "key_reason": s.key_reason,
                }
                for s in self.signals
            ],
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
            "neutral_count": self.neutral_count,
            "confidence": self.confidence,
        }


def extract_direction(report: str) -> str:
    """Extract directional signal from an analyst report.

    Strategy:
        1. Look for explicit "DIRECTION: BUY/SELL/HOLD" tag
        2. Fall back to generic signal extraction via regex
        3. Default to "NEUTRAL" if nothing found

    Args:
        report: The analyst report text.

    Returns:
        One of "BUY", "SELL", "HOLD", or "NEUTRAL".
    """
    # Try explicit DIRECTION tag first
    match = _DIRECTION_RE.search(report)
    if match:
        return match.group(1).upper()

    # Fall back to generic signal extraction
    signal = extract_signal(report)
    if signal is not None:
        return signal

    return "NEUTRAL"


def extract_all_signals(reports: dict[str, str]) -> list[AnalystSignal]:
    """Extract signals from all analyst reports.

    Args:
        reports: Mapping of report key to report text. Expected keys:
                 market_report, sentiment_report, news_report,
                 fundamentals_report.

    Returns:
        List of AnalystSignal objects, one per report found.
    """
    signals: list[AnalystSignal] = []

    for analyst_name, report_key in _ANALYST_REPORT_KEYS.items():
        report_text = reports.get(report_key, "")
        if not report_text:
            logger.debug("No report found for %s (key=%s)", analyst_name, report_key)
            continue

        direction = extract_direction(report_text)
        # Extract first sentence as key reason (or truncate)
        key_reason = _extract_key_reason(report_text)

        signals.append(
            AnalystSignal(
                analyst=analyst_name,
                direction=direction,
                key_reason=key_reason,
            )
        )
        logger.debug("Extracted %s signal from %s", direction, analyst_name)

    return signals


def _extract_key_reason(report: str, max_length: int = 200) -> str:
    """Extract a short key reason from a report.

    Takes the first sentence or truncates to max_length.
    """
    # Find first sentence ending
    sentence_end = re.search(r"[.!?]\s", report)
    if sentence_end and sentence_end.end() <= max_length:
        return report[: sentence_end.end()].strip()
    if len(report) <= max_length:
        return report.strip()
    return report[:max_length].strip() + "..."


def _compute_confidence(
    buy_count: int,
    sell_count: int,
    hold_count: int,
    neutral_count: int,
) -> str:
    """Compute confidence level from signal counts.

    Rules:
        - HIGH: >= 4 signals and all non-neutral signals agree (unanimous)
        - MEDIUM: >= 3 signals and majority (> half of non-neutral) agree
        - LOW: otherwise
    """
    total = buy_count + sell_count + hold_count + neutral_count
    non_neutral = buy_count + sell_count + hold_count
    max_directional = max(buy_count, sell_count, hold_count)

    # HIGH: 4+ signals, all non-neutral agree
    if total >= 4 and non_neutral > 0 and max_directional == non_neutral:
        return "HIGH"

    # MEDIUM: 3+ signals, majority of non-neutral agree
    if total >= 3 and non_neutral > 0 and max_directional > non_neutral / 2:
        return "MEDIUM"

    return "LOW"


def compute_consensus(signals: list[AnalystSignal]) -> AnalystConsensus:
    """Compute consensus from a list of analyst signals.

    Args:
        signals: List of AnalystSignal objects.

    Returns:
        AnalystConsensus with aggregated counts and confidence.
    """
    buy_count = sum(1 for s in signals if s.direction == "BUY")
    sell_count = sum(1 for s in signals if s.direction == "SELL")
    hold_count = sum(1 for s in signals if s.direction == "HOLD")
    neutral_count = sum(1 for s in signals if s.direction == "NEUTRAL")

    confidence = _compute_confidence(buy_count, sell_count, hold_count, neutral_count)

    return AnalystConsensus(
        signals=tuple(signals),
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
        neutral_count=neutral_count,
        confidence=confidence,
    )


def _detect_market_regime(reports: dict[str, str]) -> str:
    """Detect market regime from report content.

    Simple heuristic based on keyword frequency.
    """
    combined = " ".join(reports.values()).lower()

    bullish_keywords = ["bullish", "uptrend", "growth", "rally", "breakout", "momentum"]
    bearish_keywords = ["bearish", "downtrend", "decline", "crash", "selloff", "correction"]

    bullish_score = sum(combined.count(kw) for kw in bullish_keywords)
    bearish_score = sum(combined.count(kw) for kw in bearish_keywords)

    if bullish_score > bearish_score * 1.5:
        return "BULLISH"
    if bearish_score > bullish_score * 1.5:
        return "BEARISH"
    return "NEUTRAL"


def create_extract_signals_node():
    """Create a graph node function that extracts analyst signals.

    Returns:
        A callable suitable for use as a LangGraph node. The function
        takes state (dict) and returns a dict with analyst_consensus
        and market_regime keys.
    """

    def extract_signals_node(state: dict) -> dict:
        """Extract analyst signals and compute consensus from state."""
        reports = {
            "market_report": state.get("market_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "news_report": state.get("news_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
        }

        signals = extract_all_signals(reports)
        consensus = compute_consensus(signals)
        market_regime = _detect_market_regime(reports)

        logger.info(
            "Consensus: BUY=%d SELL=%d HOLD=%d NEUTRAL=%d confidence=%s regime=%s",
            consensus.buy_count,
            consensus.sell_count,
            consensus.hold_count,
            consensus.neutral_count,
            consensus.confidence,
            market_regime,
        )

        return {
            "analyst_consensus": consensus.to_dict,
            "market_regime": market_regime,
        }

    return extract_signals_node
