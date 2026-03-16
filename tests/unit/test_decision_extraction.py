"""Tests for tradingagents.graph.decision_extraction."""

from unittest.mock import MagicMock

import pytest

from tradingagents.exceptions import DecisionExtractionError
from tradingagents.graph.decision_extraction import (
    DecisionExtractor,
    TradingDecision,
    _validate_and_build,
)


# ---------------------------------------------------------------------------
# TradingDecision creation
# ---------------------------------------------------------------------------


class TestTradingDecision:
    def test_create_valid_decision(self):
        decision = TradingDecision(
            signal="BUY",
            confidence="HIGH",
            position_pct=0.8,
            reasoning="Strong fundamentals",
        )
        assert decision.signal == "BUY"
        assert decision.confidence == "HIGH"
        assert decision.position_pct == 0.8
        assert decision.reasoning == "Strong fundamentals"

    def test_frozen_immutability(self):
        decision = TradingDecision(
            signal="SELL",
            confidence="LOW",
            position_pct=0.2,
            reasoning="Weak outlook",
        )
        with pytest.raises(AttributeError):
            decision.signal = "BUY"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _validate_and_build
# ---------------------------------------------------------------------------


class TestValidateAndBuild:
    def test_valid_input(self):
        decision = _validate_and_build("BUY", "HIGH", 0.7, "Good reason")
        assert decision.signal == "BUY"
        assert decision.confidence == "HIGH"
        assert decision.position_pct == 0.7

    def test_case_normalization(self):
        decision = _validate_and_build("buy", "low", 0.5, "reason")
        assert decision.signal == "BUY"
        assert decision.confidence == "LOW"

    def test_clamp_position_pct_above_one(self):
        decision = _validate_and_build("BUY", "HIGH", 1.5, "reason")
        assert decision.position_pct == 1.0

    def test_clamp_position_pct_below_zero(self):
        decision = _validate_and_build("SELL", "LOW", -0.3, "reason")
        assert decision.position_pct == 0.0

    def test_position_pct_at_boundaries(self):
        d0 = _validate_and_build("HOLD", "MEDIUM", 0.0, "")
        assert d0.position_pct == 0.0
        d1 = _validate_and_build("HOLD", "MEDIUM", 1.0, "")
        assert d1.position_pct == 1.0

    def test_invalid_signal_raises(self):
        with pytest.raises(DecisionExtractionError, match="Invalid signal"):
            _validate_and_build("MAYBE", "HIGH", 0.5, "reason")

    def test_invalid_confidence_raises(self):
        with pytest.raises(DecisionExtractionError, match="Invalid confidence"):
            _validate_and_build("BUY", "VERY_HIGH", 0.5, "reason")

    def test_whitespace_stripped(self):
        decision = _validate_and_build("  BUY  ", "  HIGH  ", 0.5, "  reason  ")
        assert decision.signal == "BUY"
        assert decision.confidence == "HIGH"
        assert decision.reasoning == "reason"


# ---------------------------------------------------------------------------
# DecisionExtractor.extract
# ---------------------------------------------------------------------------


class TestDecisionExtractor:
    def _make_mock_llm(self, response_content: str) -> MagicMock:
        mock = MagicMock()
        resp = MagicMock()
        resp.content = response_content
        mock.invoke.return_value = resp
        return mock

    def test_json_direct_extraction(self):
        """JSON in the signal text is parsed directly without LLM call."""
        mock_llm = self._make_mock_llm("")
        extractor = DecisionExtractor(mock_llm)

        signal = (
            'Here is my analysis: {"signal": "BUY", "confidence": "HIGH", '
            '"position_pct": 0.75, "reasoning": "Strong momentum"}'
        )
        decision = extractor.extract(signal)
        assert decision.signal == "BUY"
        assert decision.confidence == "HIGH"
        assert decision.position_pct == 0.75
        # LLM should not be called when JSON parses directly
        mock_llm.invoke.assert_not_called()

    def test_llm_extraction_fallback(self):
        """When no JSON in text, LLM is asked to produce structured output."""
        json_response = (
            '{"signal": "SELL", "confidence": "MEDIUM", '
            '"position_pct": 0.6, "reasoning": "Declining trend"}'
        )
        mock_llm = self._make_mock_llm(json_response)
        extractor = DecisionExtractor(mock_llm)

        decision = extractor.extract("Market shows weakening indicators")
        assert decision.signal == "SELL"
        assert decision.confidence == "MEDIUM"

    def test_signal_processor_fallback(self):
        """When LLM JSON extraction also fails, falls back to SignalProcessor."""
        mock_llm = self._make_mock_llm("I think you should BUY")
        extractor = DecisionExtractor(mock_llm)

        # Input has no JSON, LLM returns non-JSON with BUY
        # _try_json_parse fails on input, _try_llm_extraction -> LLM returns
        # "I think you should BUY" -> _try_json_parse on that also fails
        # -> _try_signal_fallback -> SignalProcessor finds BUY in input?
        # Actually SignalProcessor.process_signal gets the original text.
        # Original text: "unclear analysis" has no signal, but LLM L1 finds BUY
        # Let's use text that has BUY buried for the signal processor.

        # Simpler: LLM returns something without valid JSON but with BUY keyword
        decision = extractor.extract("Our recommendation is BUY the stock now")
        # This actually parses at L1 JSON (no JSON) -> L2 LLM returns
        # "I think you should BUY" (no JSON) -> L3 signal fallback on
        # original "Our recommendation is BUY..." -> regex finds BUY
        assert decision.signal == "BUY"

    def test_confidence_override_from_consensus(self):
        """Consensus confidence overrides the extracted confidence."""
        mock_llm = self._make_mock_llm("")
        extractor = DecisionExtractor(mock_llm)

        signal = (
            '{"signal": "BUY", "confidence": "LOW", '
            '"position_pct": 0.5, "reasoning": "test"}'
        )
        consensus = {"confidence": "HIGH"}
        decision = extractor.extract(signal, analyst_consensus=consensus)
        assert decision.confidence == "HIGH"

    def test_low_confidence_buy_position_cap(self):
        """LOW confidence BUY caps position_pct to 0.3."""
        mock_llm = self._make_mock_llm("")
        extractor = DecisionExtractor(mock_llm)

        signal = (
            '{"signal": "BUY", "confidence": "LOW", '
            '"position_pct": 0.8, "reasoning": "uncertain"}'
        )
        decision = extractor.extract(signal)
        assert decision.position_pct == 0.3
        assert decision.confidence == "LOW"

    def test_low_confidence_sell_no_cap(self):
        """LOW confidence SELL does NOT cap position_pct."""
        mock_llm = self._make_mock_llm("")
        extractor = DecisionExtractor(mock_llm)

        signal = (
            '{"signal": "SELL", "confidence": "LOW", '
            '"position_pct": 0.8, "reasoning": "cautious"}'
        )
        decision = extractor.extract(signal)
        assert decision.position_pct == 0.8

    def test_high_confidence_buy_no_cap(self):
        """HIGH confidence BUY does NOT cap position_pct."""
        mock_llm = self._make_mock_llm("")
        extractor = DecisionExtractor(mock_llm)

        signal = (
            '{"signal": "BUY", "confidence": "HIGH", '
            '"position_pct": 0.9, "reasoning": "strong signal"}'
        )
        decision = extractor.extract(signal)
        assert decision.position_pct == 0.9

    def test_consensus_override_then_position_cap(self):
        """Consensus overrides confidence to LOW, then position cap applies."""
        mock_llm = self._make_mock_llm("")
        extractor = DecisionExtractor(mock_llm)

        signal = (
            '{"signal": "BUY", "confidence": "HIGH", '
            '"position_pct": 0.8, "reasoning": "test"}'
        )
        consensus = {"confidence": "LOW"}
        decision = extractor.extract(signal, analyst_consensus=consensus)
        assert decision.confidence == "LOW"
        assert decision.position_pct == 0.3

    def test_extraction_failure_raises(self):
        """When all extraction methods fail, raises DecisionExtractionError."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")

        extractor = DecisionExtractor(mock_llm)
        with pytest.raises(DecisionExtractionError):
            extractor.extract("completely unintelligible text with no keywords")
