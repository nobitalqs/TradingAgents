"""Integration tests: analyst signals → consensus → signal processing pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.exceptions import SignalProcessingError
from tradingagents.graph.analyst_signals import (
    AnalystConsensus,
    AnalystSignal,
    compute_consensus,
    create_extract_signals_node,
    extract_all_signals,
    extract_direction,
)
from tradingagents.graph.signal_processing import SignalProcessor, extract_signal


class TestAnalystSignalExtraction:
    """extract_direction → extract_all_signals → compute_consensus pipeline."""

    def test_direction_tag_extraction(self):
        report = "Analysis complete.\nDIRECTION: BUY\nStrong growth expected."
        assert extract_direction(report) == "BUY"

    def test_fallback_to_regex(self):
        report = "We recommend to SELL given the weak outlook."
        assert extract_direction(report) == "SELL"

    def test_neutral_when_no_signal(self):
        report = "Insufficient data to determine a clear direction."
        assert extract_direction(report) == "NEUTRAL"

    def test_extract_all_signals_from_full_state(self):
        reports = {
            "market_report": "DIRECTION: BUY\nBullish momentum detected.",
            "sentiment_report": "Social sentiment is overwhelmingly positive. BUY.",
            "news_report": "DIRECTION: HOLD\nMixed macro signals.",
            "fundamentals_report": "Revenue growth impressive. DIRECTION: BUY",
        }
        signals = extract_all_signals(reports)
        assert len(signals) == 4

        directions = {s.analyst: s.direction for s in signals}
        assert directions["market_analyst"] == "BUY"
        assert directions["sentiment_analyst"] == "BUY"
        assert directions["news_analyst"] == "HOLD"
        assert directions["fundamentals_analyst"] == "BUY"

    def test_missing_report_skipped(self):
        """Empty or missing reports don't produce signals."""
        reports = {
            "market_report": "DIRECTION: SELL\nWeak technicals.",
            "sentiment_report": "",
            # news_report and fundamentals_report missing entirely
        }
        signals = extract_all_signals(reports)
        assert len(signals) == 1
        assert signals[0].analyst == "market_analyst"


class TestConsensusComputation:
    """compute_consensus with various vote distributions."""

    def _make_signals(self, directions: list[str]) -> list[AnalystSignal]:
        names = ["market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst"]
        return [
            AnalystSignal(analyst=names[i], direction=d, key_reason=f"reason {i}")
            for i, d in enumerate(directions)
        ]

    def test_unanimous_buy_high_confidence(self):
        signals = self._make_signals(["BUY", "BUY", "BUY", "BUY"])
        consensus = compute_consensus(signals)
        assert consensus.buy_count == 4
        assert consensus.confidence == "HIGH"

    def test_majority_buy_medium_confidence(self):
        signals = self._make_signals(["BUY", "BUY", "SELL", "BUY"])
        consensus = compute_consensus(signals)
        assert consensus.buy_count == 3
        assert consensus.sell_count == 1
        assert consensus.confidence == "MEDIUM"

    def test_split_vote_low_confidence(self):
        signals = self._make_signals(["BUY", "SELL"])
        consensus = compute_consensus(signals[:2])
        assert consensus.confidence == "LOW"

    def test_consensus_to_dict_serialization(self):
        signals = self._make_signals(["BUY", "BUY", "HOLD", "NEUTRAL"])
        consensus = compute_consensus(signals)
        d = consensus.to_dict
        assert d["buy_count"] == 2
        assert d["hold_count"] == 1
        assert d["neutral_count"] == 1
        assert len(d["signals"]) == 4


class TestExtractSignalsNode:
    """create_extract_signals_node produces a graph-compatible node."""

    def test_node_extracts_consensus_and_regime(self):
        node_fn = create_extract_signals_node()
        state = {
            "market_report": "DIRECTION: BUY\nBullish rally underway with strong momentum.",
            "sentiment_report": "DIRECTION: BUY\nPositive social buzz and uptrend.",
            "news_report": "DIRECTION: SELL\nBearish macro environment and correction likely.",
            "fundamentals_report": "DIRECTION: BUY\nExcellent revenue growth and breakout.",
        }
        result = node_fn(state)

        assert "analyst_consensus" in result
        assert "market_regime" in result

        consensus = result["analyst_consensus"]
        assert consensus["buy_count"] == 3
        assert consensus["sell_count"] == 1

    def test_node_detects_bearish_regime(self):
        node_fn = create_extract_signals_node()
        state = {
            "market_report": "Bearish downtrend with selloff and crash risk.",
            "sentiment_report": "Bearish decline expected.",
            "news_report": "Correction is imminent, bearish outlook.",
            "fundamentals_report": "Revenue decline and bearish trend.",
        }
        result = node_fn(state)
        assert result["market_regime"] == "BEARISH"


class TestSignalProcessorPipeline:
    """SignalProcessor L1→L2→L3 fallback chain."""

    def test_l1_regex_extraction(self):
        """Clear BUY/SELL/HOLD in text → L1 handles it."""
        processor = SignalProcessor(llm=None)
        assert processor.process_signal("Final recommendation: BUY") == "BUY"
        assert processor.process_signal("We advise to SELL") == "SELL"
        assert processor.process_signal("Maintain HOLD position") == "HOLD"

    def test_l2_fallback_when_regex_fails(self):
        """No regex match → L2 uses LLM → regex on LLM output."""
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "Based on analysis, the decision is BUY."
        mock_llm.invoke.return_value = response

        processor = SignalProcessor(llm=mock_llm)
        # Input with no BUY/SELL/HOLD keywords
        result = processor.process_signal("the outlook is positive and growth-oriented")
        assert result == "BUY"
        assert mock_llm.invoke.call_count == 1

    def test_l3_strict_fallback(self):
        """L1 fails, L2 LLM returns garbage, L3 returns exact signal."""
        mock_llm = MagicMock()

        # L2 response has no signal keyword
        l2_resp = MagicMock()
        l2_resp.content = "unclear direction"
        # L3 response is exactly "SELL"
        l3_resp = MagicMock()
        l3_resp.content = "SELL"
        mock_llm.invoke.side_effect = [l2_resp, l3_resp]

        processor = SignalProcessor(llm=mock_llm)
        result = processor.process_signal("ambiguous market conditions ahead")
        assert result == "SELL"
        assert mock_llm.invoke.call_count == 2

    def test_all_levels_fail_raises(self):
        """All levels fail → SignalProcessingError."""
        mock_llm = MagicMock()
        bad_resp = MagicMock()
        bad_resp.content = "I'm not sure"
        mock_llm.invoke.return_value = bad_resp

        processor = SignalProcessor(llm=mock_llm)
        with pytest.raises(SignalProcessingError):
            processor.process_signal("no clear direction in this text")

    def test_no_llm_skips_to_error(self):
        """No LLM configured → L2/L3 skipped → raises if L1 fails."""
        processor = SignalProcessor(llm=None)
        with pytest.raises(SignalProcessingError):
            processor.process_signal("ambiguous text without any keywords")


class TestEndToEndSignalFlow:
    """Full flow: reports → node → signal processor."""

    def test_reports_to_final_signal(self):
        """Simulate: reports → extract node → consensus → signal processor."""
        node_fn = create_extract_signals_node()
        state = {
            "market_report": "DIRECTION: BUY\nStrong uptrend.",
            "sentiment_report": "DIRECTION: BUY\nBullish sentiment.",
            "news_report": "DIRECTION: BUY\nPositive earnings.",
            "fundamentals_report": "DIRECTION: BUY\nRevenue beat.",
        }

        result = node_fn(state)
        consensus = result["analyst_consensus"]

        # Simulate final decision text
        decision_text = (
            f"Based on unanimous consensus ({consensus['buy_count']} BUY), "
            f"the final recommendation is BUY."
        )

        processor = SignalProcessor(llm=None)
        signal = processor.process_signal(decision_text)
        assert signal == "BUY"
