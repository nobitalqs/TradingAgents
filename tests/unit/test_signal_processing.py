"""Tests for tradingagents.graph.signal_processing."""

from unittest.mock import MagicMock

import pytest

from tradingagents.exceptions import SignalProcessingError
from tradingagents.graph.signal_processing import SignalProcessor, extract_signal


# ---------------------------------------------------------------------------
# extract_signal
# ---------------------------------------------------------------------------


class TestExtractSignal:
    def test_buy_uppercase(self):
        assert extract_signal("The recommendation is BUY") == "BUY"

    def test_sell_uppercase(self):
        assert extract_signal("We suggest SELL immediately") == "SELL"

    def test_hold_uppercase(self):
        assert extract_signal("Our stance is HOLD") == "HOLD"

    def test_case_insensitive_buy(self):
        assert extract_signal("I think we should buy the stock") == "BUY"

    def test_case_insensitive_sell(self):
        assert extract_signal("analysts say sell") == "SELL"

    def test_case_insensitive_hold(self):
        assert extract_signal("recommendation: hold for now") == "HOLD"

    def test_mixed_case(self):
        assert extract_signal("final call: Hold") == "HOLD"

    def test_garbage_returns_none(self):
        assert extract_signal("no signal here at all") is None

    def test_empty_string_returns_none(self):
        assert extract_signal("") is None

    def test_last_occurrence_wins(self):
        text = "Initially BUY, but after review SELL, final decision HOLD"
        assert extract_signal(text) == "HOLD"

    def test_last_occurrence_buy_after_hold(self):
        text = "HOLD the position... actually BUY more"
        assert extract_signal(text) == "BUY"

    def test_signal_in_longer_word_not_matched(self):
        # "BUYING" should not match because \b requires word boundary
        assert extract_signal("BUYING pressure is high") is None

    def test_signal_surrounded_by_punctuation(self):
        assert extract_signal("Decision: BUY.") == "BUY"


# ---------------------------------------------------------------------------
# SignalProcessor.process_signal
# ---------------------------------------------------------------------------


class TestSignalProcessor:
    def test_direct_regex_extraction(self):
        """L1: Text containing BUY/SELL/HOLD extracts directly."""
        processor = SignalProcessor()
        assert processor.process_signal("Our final decision is BUY") == "BUY"

    def test_l2_llm_extraction(self):
        """L2: When regex fails, LLM response is parsed with regex."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Based on my analysis, the decision is SELL."
        mock_llm.invoke.return_value = mock_response

        processor = SignalProcessor(llm=mock_llm)
        # Input has no BUY/SELL/HOLD so L1 fails
        result = processor.process_signal("the analysis suggests going short")
        assert result == "SELL"
        mock_llm.invoke.assert_called_once()

    def test_l3_strict_llm(self):
        """L3: Strict LLM returns exactly BUY/SELL/HOLD."""
        call_count = 0

        mock_llm = MagicMock()

        def invoke_side_effect(messages):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count == 1:
                # L2: returns garbage (no BUY/SELL/HOLD)
                resp.content = "I'm not sure about this one"
            else:
                # L3: returns exact signal
                resp.content = "HOLD"
            return resp

        mock_llm.invoke.side_effect = invoke_side_effect

        processor = SignalProcessor(llm=mock_llm)
        result = processor.process_signal("ambiguous analysis with no clear direction")
        assert result == "HOLD"
        assert mock_llm.invoke.call_count == 2

    def test_all_levels_fail_raises_error(self):
        """L4: When all levels fail, SignalProcessingError is raised."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I cannot determine a signal"
        mock_llm.invoke.return_value = mock_response

        processor = SignalProcessor(llm=mock_llm)
        with pytest.raises(SignalProcessingError):
            processor.process_signal("completely ambiguous text with no keywords")

    def test_no_llm_empty_text_raises_error(self):
        """No LLM + no signal in text -> error."""
        processor = SignalProcessor()
        with pytest.raises(SignalProcessingError):
            processor.process_signal("nothing useful here")

    def test_l2_llm_exception_falls_through(self):
        """If LLM raises an exception in L2, processing continues to L3/L4."""
        call_count = 0
        mock_llm = MagicMock()

        def invoke_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM connection failed")
            resp = MagicMock()
            resp.content = "BUY"
            return resp

        mock_llm.invoke.side_effect = invoke_side_effect

        processor = SignalProcessor(llm=mock_llm)
        result = processor.process_signal("some unclear analysis")
        assert result == "BUY"
