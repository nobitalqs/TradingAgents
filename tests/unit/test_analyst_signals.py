"""Tests for tradingagents.graph.analyst_signals."""

import pytest

from tradingagents.graph.analyst_signals import (
    AnalystConsensus,
    AnalystSignal,
    compute_consensus,
    create_extract_signals_node,
    extract_all_signals,
    extract_direction,
)


# ---------------------------------------------------------------------------
# extract_direction
# ---------------------------------------------------------------------------


class TestExtractDirection:
    def test_with_direction_tag_buy(self):
        report = "Analysis complete.\nDIRECTION: BUY\nStrong fundamentals."
        assert extract_direction(report) == "BUY"

    def test_with_direction_tag_sell(self):
        report = "DIRECTION: SELL\nWeakening outlook."
        assert extract_direction(report) == "SELL"

    def test_with_direction_tag_hold(self):
        report = "DIRECTION: hold\nNo clear trend."
        assert extract_direction(report) == "HOLD"

    def test_without_direction_tag_fallback_to_regex(self):
        report = "We recommend a BUY position on the stock."
        assert extract_direction(report) == "BUY"

    def test_without_direction_tag_sell(self):
        report = "The best action is to SELL at this price."
        assert extract_direction(report) == "SELL"

    def test_garbage_returns_neutral(self):
        report = "The market is doing interesting things with no clear signal."
        assert extract_direction(report) == "NEUTRAL"

    def test_empty_string_returns_neutral(self):
        assert extract_direction("") == "NEUTRAL"

    def test_direction_tag_takes_precedence(self):
        """DIRECTION tag overrides inline BUY/SELL/HOLD mentions."""
        report = "We see BUY signals everywhere.\nDIRECTION: SELL\nBut risk is high."
        assert extract_direction(report) == "SELL"


# ---------------------------------------------------------------------------
# compute_consensus
# ---------------------------------------------------------------------------


class TestComputeConsensus:
    def _make_signal(self, direction: str, analyst: str = "test") -> AnalystSignal:
        return AnalystSignal(analyst=analyst, direction=direction, key_reason="reason")

    def test_four_buy_signals_high_confidence(self):
        signals = [self._make_signal("BUY", f"a{i}") for i in range(4)]
        consensus = compute_consensus(signals)
        assert consensus.confidence == "HIGH"
        assert consensus.buy_count == 4
        assert consensus.sell_count == 0
        assert consensus.hold_count == 0
        assert consensus.neutral_count == 0

    def test_four_sell_signals_high_confidence(self):
        signals = [self._make_signal("SELL", f"a{i}") for i in range(4)]
        consensus = compute_consensus(signals)
        assert consensus.confidence == "HIGH"
        assert consensus.sell_count == 4

    def test_three_buy_one_sell_medium_confidence(self):
        signals = [
            self._make_signal("BUY", "a0"),
            self._make_signal("BUY", "a1"),
            self._make_signal("BUY", "a2"),
            self._make_signal("SELL", "a3"),
        ]
        consensus = compute_consensus(signals)
        assert consensus.confidence == "MEDIUM"
        assert consensus.buy_count == 3
        assert consensus.sell_count == 1

    def test_split_signals_low_confidence(self):
        signals = [
            self._make_signal("BUY", "a0"),
            self._make_signal("SELL", "a1"),
            self._make_signal("HOLD", "a2"),
            self._make_signal("NEUTRAL", "a3"),
        ]
        consensus = compute_consensus(signals)
        assert consensus.confidence == "LOW"

    def test_two_buy_two_sell_low_confidence(self):
        signals = [
            self._make_signal("BUY", "a0"),
            self._make_signal("BUY", "a1"),
            self._make_signal("SELL", "a2"),
            self._make_signal("SELL", "a3"),
        ]
        consensus = compute_consensus(signals)
        assert consensus.confidence == "LOW"

    def test_three_buy_one_neutral_medium(self):
        """3 BUY + 1 NEUTRAL -> total=4, non_neutral=3, max_dir=3 == non_neutral -> HIGH."""
        signals = [
            self._make_signal("BUY", "a0"),
            self._make_signal("BUY", "a1"),
            self._make_signal("BUY", "a2"),
            self._make_signal("NEUTRAL", "a3"),
        ]
        consensus = compute_consensus(signals)
        # 4 total, 3 non-neutral all BUY => unanimous => HIGH
        assert consensus.confidence == "HIGH"

    def test_empty_signals_low_confidence(self):
        consensus = compute_consensus([])
        assert consensus.confidence == "LOW"
        assert consensus.buy_count == 0

    def test_signals_tuple_is_immutable(self):
        signals = [self._make_signal("BUY")]
        consensus = compute_consensus(signals)
        assert isinstance(consensus.signals, tuple)

    def test_to_dict_property(self):
        signals = [self._make_signal("BUY", "market_analyst")]
        consensus = compute_consensus(signals)
        d = consensus.to_dict
        assert isinstance(d, dict)
        assert d["buy_count"] == 1
        assert d["confidence"] == "LOW"
        assert len(d["signals"]) == 1
        assert d["signals"][0]["analyst"] == "market_analyst"
        assert d["signals"][0]["direction"] == "BUY"


# ---------------------------------------------------------------------------
# extract_all_signals
# ---------------------------------------------------------------------------


class TestExtractAllSignals:
    def test_all_four_reports(self):
        reports = {
            "market_report": "DIRECTION: BUY\nBullish trend.",
            "sentiment_report": "DIRECTION: SELL\nNegative sentiment.",
            "news_report": "DIRECTION: HOLD\nMixed news.",
            "fundamentals_report": "DIRECTION: BUY\nStrong earnings.",
        }
        signals = extract_all_signals(reports)
        assert len(signals) == 4

        directions = {s.analyst: s.direction for s in signals}
        assert directions["market_analyst"] == "BUY"
        assert directions["sentiment_analyst"] == "SELL"
        assert directions["news_analyst"] == "HOLD"
        assert directions["fundamentals_analyst"] == "BUY"

    def test_missing_reports_skipped(self):
        reports = {
            "market_report": "DIRECTION: BUY\nGood outlook.",
        }
        signals = extract_all_signals(reports)
        assert len(signals) == 1
        assert signals[0].analyst == "market_analyst"

    def test_empty_reports(self):
        signals = extract_all_signals({})
        assert len(signals) == 0

    def test_empty_string_report_skipped(self):
        reports = {
            "market_report": "",
            "sentiment_report": "DIRECTION: SELL\nBad vibes.",
        }
        signals = extract_all_signals(reports)
        assert len(signals) == 1
        assert signals[0].analyst == "sentiment_analyst"

    def test_key_reason_populated(self):
        reports = {
            "market_report": "Strong upward momentum. Multiple indicators confirm.",
        }
        signals = extract_all_signals(reports)
        assert len(signals) == 1
        assert len(signals[0].key_reason) > 0


# ---------------------------------------------------------------------------
# create_extract_signals_node
# ---------------------------------------------------------------------------


class TestExtractSignalsNode:
    def test_node_returns_expected_keys(self):
        node_fn = create_extract_signals_node()
        state = {
            "market_report": "DIRECTION: BUY\nBullish.",
            "sentiment_report": "DIRECTION: BUY\nPositive.",
            "news_report": "DIRECTION: BUY\nGood news.",
            "fundamentals_report": "DIRECTION: BUY\nStrong earnings.",
        }
        result = node_fn(state)
        assert "analyst_consensus" in result
        assert "market_regime" in result
        assert result["analyst_consensus"]["confidence"] == "HIGH"
        assert result["analyst_consensus"]["buy_count"] == 4

    def test_node_handles_empty_state(self):
        node_fn = create_extract_signals_node()
        result = node_fn({})
        assert result["analyst_consensus"]["buy_count"] == 0
        assert result["market_regime"] == "NEUTRAL"
