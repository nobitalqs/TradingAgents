"""Unit tests for AutoReflector — T+N reflection on trading decisions."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.learning.auto_reflect import AutoReflector
from tradingagents.learning.persistence import MemoryStore


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def memory_store(tmp_path):
    return MemoryStore(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def mock_reflector():
    llm = MagicMock()
    resp = MagicMock()
    resp.content = "Key insight: position was well-timed given momentum."
    llm.invoke.return_value = resp

    from tradingagents.graph.reflection import Reflector

    return Reflector(llm=llm)


@pytest.fixture
def memories():
    from tradingagents.agents.utils.memory import FinancialSituationMemory

    return {
        "bull_memory": FinancialSituationMemory("bull", {}),
        "bear_memory": FinancialSituationMemory("bear", {}),
        "trader_memory": FinancialSituationMemory("trader", {}),
        "invest_judge_memory": FinancialSituationMemory("invest_judge", {}),
        "risk_manager_memory": FinancialSituationMemory("risk_judge", {}),
    }


@pytest.fixture
def sample_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "market_report": "Strong bullish momentum with breakout.",
        "sentiment_report": "Positive social media buzz.",
        "news_report": "Earnings beat expectations.",
        "fundamentals_report": "Revenue up 20% YoY.",
        "investment_debate_state": {
            "bull_history": ["Growth expected."],
            "bear_history": ["Overvaluation risk."],
            "judge_decision": "Proceed with BUY.",
        },
        "trader_investment_plan": "Buy at market open.",
        "risk_debate_state": {"judge_decision": "Moderate risk."},
        "final_trade_decision": "BUY 100 shares NVDA.",
    }


def _make_price_df(dates_prices: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a DataFrame mimicking yfinance output."""
    dates = [pd.Timestamp(d) for d, _ in dates_prices]
    prices = [p for _, p in dates_prices]
    return pd.DataFrame({"Close": prices}, index=pd.DatetimeIndex(dates))


# ── Direction checking ───────────────────────────────────────────────


class TestCheckDirection:
    def test_buy_price_up_correct(self):
        assert AutoReflector._check_direction("BUY", 5.0) is True

    def test_buy_price_down_incorrect(self):
        assert AutoReflector._check_direction("BUY", -3.0) is False

    def test_sell_price_down_correct(self):
        assert AutoReflector._check_direction("SELL", -2.0) is True

    def test_sell_price_up_incorrect(self):
        assert AutoReflector._check_direction("SELL", 4.0) is False

    def test_hold_returns_none(self):
        assert AutoReflector._check_direction("HOLD", 1.0) is None

    def test_case_insensitive(self):
        assert AutoReflector._check_direction("buy", 1.0) is True
        assert AutoReflector._check_direction("Sell", -1.0) is True


# ── Returns description ──────────────────────────────────────────────


class TestDescribeReturns:
    def test_correct_buy(self):
        desc = AutoReflector._describe_returns("NVDA", "BUY", 100.0, 105.0, 5.0, True)
        assert "NVDA" in desc
        assert "BUY" in desc
        assert "+5.00%" in desc
        assert "CORRECT" in desc

    def test_incorrect_sell(self):
        desc = AutoReflector._describe_returns("AAPL", "SELL", 200.0, 210.0, 5.0, False)
        assert "INCORRECT" in desc
        assert "UP" in desc

    def test_hold_neutral(self):
        desc = AutoReflector._describe_returns("GOOG", "HOLD", 150.0, 148.0, -1.33, None)
        assert "NEUTRAL" in desc


# ── State loading ────────────────────────────────────────────────────


class TestLoadState:
    def test_valid_json(self):
        state = AutoReflector._load_state('{"market_report": "bullish"}')
        assert state == {"market_report": "bullish"}

    def test_empty_string(self):
        assert AutoReflector._load_state("") is None

    def test_invalid_json(self):
        assert AutoReflector._load_state("not json") is None


# ── Price fetching ───────────────────────────────────────────────────


class TestFetchPrices:
    def test_find_closest_price_exact(self):
        df = _make_price_df([("2026-01-15", 150.0), ("2026-01-16", 152.0)])
        target = datetime(2026, 1, 15)
        assert AutoReflector._find_closest_price(df, target) == 150.0

    def test_find_closest_price_weekend(self):
        """Saturday falls back to Friday."""
        df = _make_price_df([("2026-01-17", 150.0)])  # Friday
        target = datetime(2026, 1, 18)  # Saturday
        assert AutoReflector._find_closest_price(df, target) == 150.0

    def test_find_closest_price_none(self):
        df = _make_price_df([("2026-01-10", 100.0)])
        target = datetime(2026, 1, 20)  # Too far
        assert AutoReflector._find_closest_price(df, target) is None

    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_fetch_prices_success(self, mock_download):
        df = _make_price_df([
            ("2026-01-15", 100.0),
            ("2026-01-16", 101.0),
            ("2026-01-22", 105.0),
        ])
        mock_download.return_value = df

        ar = AutoReflector.__new__(AutoReflector)
        result = ar._fetch_prices("NVDA", "2026-01-15", 7)
        assert result is not None
        assert result[0] == 100.0
        assert result[1] == 105.0

    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_fetch_prices_empty_data(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        ar = AutoReflector.__new__(AutoReflector)
        result = ar._fetch_prices("NVDA", "2026-01-15", 7)
        assert result is None

    def test_fetch_prices_invalid_date(self):
        ar = AutoReflector.__new__(AutoReflector)
        result = ar._fetch_prices("NVDA", "not-a-date", 7)
        assert result is None


# ── Full reflect ─────────────────────────────────────────────────────


class TestReflect:
    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_reflect_single_record(
        self, mock_download, mock_reflector, memory_store, memories, sample_state
    ):
        df = _make_price_df([
            ("2026-01-15", 100.0),
            ("2026-01-22", 108.0),
        ])
        mock_download.return_value = df

        ar = AutoReflector(mock_reflector, memory_store, memories, horizon=7)
        result = ar.reflect("NVDA", "2026-01-15", "BUY", sample_state)

        assert result is not None
        assert result["direction_correct"] is True
        assert result["price_change_pct"] == 8.0
        assert result["price_t0"] == 100.0
        assert result["price_tn"] == 108.0

        # Memories should have been populated by the 5 reflectors
        for mem in memories.values():
            assert len(mem.documents) > 0

        # DB should be updated
        pending = memory_store.get_pending_reflections(horizon_days=0, as_of_date="2030-01-01")
        assert len(pending) == 0  # no records (nothing was saved pre-reflect)

    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_reflect_returns_none_on_price_failure(
        self, mock_download, mock_reflector, memory_store, memories, sample_state
    ):
        mock_download.return_value = pd.DataFrame()

        ar = AutoReflector(mock_reflector, memory_store, memories, horizon=7)
        result = ar.reflect("NVDA", "2026-01-15", "BUY", sample_state)
        assert result is None


# ── Batch reflect_pending ────────────────────────────────────────────


class TestReflectPending:
    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_reflect_pending_processes_records(
        self, mock_download, mock_reflector, memory_store, memories, sample_state
    ):
        # Save a pending record
        memory_store.save_analysis_result(
            ticker="NVDA",
            trade_date="2026-01-01",
            signal="BUY",
            confidence="HIGH",
            full_decision="BUY 100 shares",
            state_json=json.dumps(sample_state),
        )

        df = _make_price_df([
            ("2025-12-31", 95.0),
            ("2026-01-01", 100.0),
            ("2026-01-08", 110.0),
        ])
        mock_download.return_value = df

        ar = AutoReflector(mock_reflector, memory_store, memories, horizon=7)
        results = ar.reflect_pending(as_of_date="2026-01-15")

        assert len(results) == 1
        assert results[0]["ticker"] == "NVDA"
        assert results[0]["direction_correct"] is True
        assert results[0]["price_change_pct"] == 10.0

        # Record should now be reflected
        pending = memory_store.get_pending_reflections(
            horizon_days=0, as_of_date="2030-01-01"
        )
        assert len(pending) == 0

    def test_reflect_pending_no_records(
        self, mock_reflector, memory_store, memories
    ):
        ar = AutoReflector(mock_reflector, memory_store, memories, horizon=7)
        results = ar.reflect_pending(as_of_date="2026-01-15")
        assert results == []

    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_reflect_pending_skips_no_state(
        self, mock_download, mock_reflector, memory_store, memories
    ):
        # Save without state_json
        memory_store.save_analysis_result(
            ticker="AAPL",
            trade_date="2026-01-01",
            signal="SELL",
            confidence="LOW",
            full_decision="SELL",
            state_json="",
        )

        ar = AutoReflector(mock_reflector, memory_store, memories, horizon=7)
        results = ar.reflect_pending(as_of_date="2026-01-15")
        assert results == []

    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_reflect_pending_respects_horizon(
        self, mock_download, mock_reflector, memory_store, memories, sample_state
    ):
        # Trade on Jan 10 — with 7-day horizon, only eligible after Jan 17
        memory_store.save_analysis_result(
            ticker="TSLA",
            trade_date="2026-01-10",
            signal="BUY",
            confidence="MEDIUM",
            full_decision="BUY TSLA",
            state_json=json.dumps(sample_state),
        )

        ar = AutoReflector(mock_reflector, memory_store, memories, horizon=7)

        # Too early
        results = ar.reflect_pending(as_of_date="2026-01-15")
        assert results == []

        # Now eligible
        df = _make_price_df([
            ("2026-01-10", 200.0),
            ("2026-01-17", 190.0),
        ])
        mock_download.return_value = df

        results = ar.reflect_pending(as_of_date="2026-01-20")
        assert len(results) == 1
        assert results[0]["direction_correct"] is False  # BUY but price dropped


# ── Memory persistence ───────────────────────────────────────────────


class TestMemoryPersistence:
    @patch("tradingagents.learning.auto_reflect.yf.download")
    def test_memories_persisted_to_sqlite(
        self, mock_download, mock_reflector, memory_store, memories, sample_state
    ):
        df = _make_price_df([
            ("2026-01-15", 100.0),
            ("2026-01-22", 105.0),
        ])
        mock_download.return_value = df

        ar = AutoReflector(mock_reflector, memory_store, memories, horizon=7)
        ar.reflect("NVDA", "2026-01-15", "BUY", sample_state)

        # Each of the 5 memories should have been persisted
        total = sum(
            memory_store.get_memory_count(name)
            for name in [
                "bull_memory", "bear_memory", "trader_memory",
                "invest_judge_memory", "risk_manager_memory",
            ]
        )
        assert total >= 5
