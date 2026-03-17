"""Integration tests: MemoryStore + Reflector → persist reflection results."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.agent_states import (
    create_empty_invest_debate_state,
    create_empty_risk_debate_state,
)
from tradingagents.graph.reflection import Reflector, reflect_memories
from tradingagents.learning.persistence import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    """Fresh SQLite MemoryStore in a temp directory."""
    db = tmp_path / "test_memory.db"
    return MemoryStore(db_path=str(db))


@pytest.fixture
def mock_reflector():
    """Reflector with a mock LLM that returns deterministic reflections."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Key insight: position sizing was correct given the market conditions."
    llm.invoke.return_value = response
    return Reflector(llm=llm)


@pytest.fixture
def rich_state():
    """AgentState with filled reports and debate history."""
    invest_debate = create_empty_invest_debate_state()
    invest_debate["bull_history"] = ["Strong growth expected in Q1."]
    invest_debate["bear_history"] = ["Overvaluation concerns persist."]
    invest_debate["judge_decision"] = "Proceed with caution, BUY small."

    risk_debate = create_empty_risk_debate_state()
    risk_debate["judge_decision"] = "Risk is moderate, position size appropriate."

    return {
        "messages": [],
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "sender": "",
        "market_report": "Technical indicators show bullish momentum with breakout patterns.",
        "sentiment_report": "Social media sentiment is overwhelmingly positive for NVDA.",
        "news_report": "NVDA announced record revenue in latest earnings call.",
        "fundamentals_report": "Revenue up 20% YoY, strong free cash flow.",
        "analyst_consensus": {},
        "market_regime": {},
        "data_credibility": {},
        "investment_debate_state": invest_debate,
        "investment_plan": "Buy NVDA with 5% portfolio allocation.",
        "trader_investment_plan": "Execute at market open with trailing stop.",
        "risk_debate_state": risk_debate,
        "final_trade_decision": "BUY 100 shares NVDA at market open.",
    }


class TestMemoryStorePersistence:
    """MemoryStore CRUD operations across tables."""

    def test_save_and_load_memories(self, memory_store):
        situations = [
            ("bullish market with earnings beat", "increase position size"),
            ("bearish market with rate hike", "reduce exposure"),
        ]
        memory_store.save_memories("bull_memory", situations, ticker="NVDA")

        loaded = memory_store.load_memories("bull_memory")
        assert len(loaded) == 2
        assert set(loaded) == set(situations)

    def test_idempotent_writes(self, memory_store):
        """Duplicate content_hash → INSERT OR IGNORE, no duplicates."""
        situation = ("same situation", "same recommendation")
        memory_store.save_memories("test", [situation])
        memory_store.save_memories("test", [situation])

        assert memory_store.get_memory_count("test") == 1

    def test_analysis_result_upsert(self, memory_store):
        memory_store.save_analysis_result(
            ticker="NVDA",
            trade_date="2026-01-15",
            signal="BUY",
            confidence="HIGH",
            full_decision="Buy 100 shares.",
        )

        # Upsert with updated signal
        memory_store.save_analysis_result(
            ticker="NVDA",
            trade_date="2026-01-15",
            signal="HOLD",
            confidence="MEDIUM",
            full_decision="Changed to hold.",
        )

        # Should still be 1 row
        pending = memory_store.get_pending_reflections(
            horizon_days=0, as_of_date="2026-01-16"
        )
        assert len(pending) == 1
        assert pending[0]["signal"] == "HOLD"

    def test_pending_reflections_respect_horizon(self, memory_store):
        memory_store.save_analysis_result(
            ticker="NVDA",
            trade_date="2026-01-01",
            signal="BUY",
            confidence="HIGH",
            full_decision="Buy early Jan.",
        )
        memory_store.save_analysis_result(
            ticker="AAPL",
            trade_date="2026-01-10",
            signal="SELL",
            confidence="LOW",
            full_decision="Sell mid Jan.",
        )

        # With horizon=7 and as_of_date=2026-01-15,
        # only trades on or before 2026-01-08 qualify
        pending = memory_store.get_pending_reflections(
            horizon_days=7, as_of_date="2026-01-15"
        )
        assert len(pending) == 1
        assert pending[0]["ticker"] == "NVDA"

    def test_update_reflection_marks_reflected(self, memory_store):
        memory_store.save_analysis_result(
            ticker="NVDA",
            trade_date="2026-01-01",
            signal="BUY",
            confidence="HIGH",
            full_decision="Buy.",
        )
        memory_store.update_reflection(
            ticker="NVDA",
            trade_date="2026-01-01",
            actual_return=0.05,
            direction_correct=True,
        )

        # No longer pending
        pending = memory_store.get_pending_reflections(
            horizon_days=0, as_of_date="2026-12-31"
        )
        assert len(pending) == 0

    def test_timing_records(self, memory_store):
        memory_store.save_timing("NVDA", "2026-01-15", 12.5)
        memory_store.save_timing("NVDA", "2026-01-15", 15.3)

        # Multiple timing records are allowed (not unique)
        # Verify no error on duplicate insert


class TestReflectorIntegration:
    """Reflector + mock LLM + mock memory cooperate correctly."""

    def test_reflect_bull_updates_memory(self, mock_reflector, rich_state):
        memory = MagicMock()
        mock_reflector.reflect_bull(rich_state, returns_losses=0.05, memory=memory)

        memory.add_situations.assert_called_once()
        args = memory.add_situations.call_args[0][0]
        assert len(args) == 1
        situation, recommendation = args[0]
        assert "bullish momentum" in situation.lower() or len(situation) > 0
        assert "insight" in recommendation.lower() or len(recommendation) > 0

    def test_reflect_memories_calls_all_five(self, mock_reflector, rich_state):
        memories = {
            "bull_memory": MagicMock(),
            "bear_memory": MagicMock(),
            "trader_memory": MagicMock(),
            "invest_judge_memory": MagicMock(),
            "risk_manager_memory": MagicMock(),
        }

        reflect_memories(mock_reflector, rich_state, returns_losses=0.03, memories=memories)

        for name, mem in memories.items():
            mem.add_situations.assert_called_once(), f"{name} was not called"

    def test_reflection_uses_state_context(self, rich_state):
        """LLM receives report content from state."""
        llm = MagicMock()
        resp = MagicMock()
        resp.content = "Reflection result."
        llm.invoke.return_value = resp

        reflector = Reflector(llm=llm)
        memory = MagicMock()
        reflector.reflect_trader(rich_state, returns_losses=-0.02, memory=memory)

        # Verify LLM was called with content from state
        call_args = llm.invoke.call_args[0][0]
        # call_args is a list of (role, content) tuples
        human_msg = call_args[1][1]
        assert "TRADER" in human_msg
        assert "-0.02" in human_msg


class TestReflectionToPersistence:
    """Full cycle: Reflector reflects → MemoryStore persists."""

    def test_reflection_saved_to_store(self, memory_store, mock_reflector, rich_state):
        """Reflections flow from Reflector into MemoryStore."""

        class PersistentMemory:
            """Adapter bridging memory.add_situations to MemoryStore."""

            def __init__(self, store: MemoryStore, name: str):
                self._store = store
                self._name = name

            def add_situations(self, situations: list[tuple[str, str]]) -> None:
                self._store.save_memories(self._name, situations, ticker="NVDA")

        memories = {
            "bull_memory": PersistentMemory(memory_store, "bull"),
            "bear_memory": PersistentMemory(memory_store, "bear"),
            "trader_memory": PersistentMemory(memory_store, "trader"),
            "invest_judge_memory": PersistentMemory(memory_store, "invest_judge"),
            "risk_manager_memory": PersistentMemory(memory_store, "risk_manager"),
        }

        reflect_memories(mock_reflector, rich_state, returns_losses=0.05, memories=memories)

        # Each component should have exactly 1 memory stored
        for name in ["bull", "bear", "trader", "invest_judge", "risk_manager"]:
            count = memory_store.get_memory_count(name)
            assert count == 1, f"{name} should have 1 memory, got {count}"

        # Verify content can be loaded back
        bull_mems = memory_store.load_memories("bull")
        assert len(bull_mems) == 1
        situation, recommendation = bull_mems[0]
        assert len(situation) > 0
        assert len(recommendation) > 0
