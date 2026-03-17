"""Tests for BM25 memory eviction and SQLite loading."""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.memory import FinancialSituationMemory


class TestEviction:
    """Oldest entries evicted when max_entries exceeded."""

    def test_under_limit_no_eviction(self):
        mem = FinancialSituationMemory("test", max_entries=5)
        mem.add_situations([("a", "r1"), ("b", "r2"), ("c", "r3")])
        assert len(mem.documents) == 3

    def test_at_limit_no_eviction(self):
        mem = FinancialSituationMemory("test", max_entries=3)
        mem.add_situations([("a", "r1"), ("b", "r2"), ("c", "r3")])
        assert len(mem.documents) == 3
        assert mem.documents == ["a", "b", "c"]

    def test_over_limit_evicts_oldest(self):
        mem = FinancialSituationMemory("test", max_entries=3)
        mem.add_situations([("a", "r1"), ("b", "r2"), ("c", "r3")])
        mem.add_situations([("d", "r4"), ("e", "r5")])
        assert len(mem.documents) == 3
        # Oldest (a, b) evicted; c, d, e remain
        assert mem.documents == ["c", "d", "e"]
        assert mem.recommendations == ["r3", "r4", "r5"]

    def test_eviction_rebuilds_index(self):
        mem = FinancialSituationMemory("test", max_entries=2)
        mem.add_situations([("alpha beta gamma", "rec1")])
        mem.add_situations([("delta epsilon", "rec2")])
        mem.add_situations([("zeta theta", "rec3")])

        # "alpha beta gamma" evicted
        assert len(mem.documents) == 2
        assert "alpha beta gamma" not in mem.documents
        # BM25 index was rebuilt and is functional
        assert mem.bm25 is not None
        results = mem.get_memories("some query", n_matches=2)
        assert len(results) == 2
        # Evicted doc should not appear in results
        matched = {r["matched_situation"] for r in results}
        assert "alpha beta gamma" not in matched

    def test_bulk_add_exceeding_limit(self):
        mem = FinancialSituationMemory("test", max_entries=2)
        mem.add_situations([("a", "r1"), ("b", "r2"), ("c", "r3"), ("d", "r4"), ("e", "r5")])
        assert len(mem.documents) == 2
        assert mem.documents == ["d", "e"]

    def test_default_max_entries(self):
        mem = FinancialSituationMemory("test")
        assert mem.max_entries == 200


class TestClearAndReload:
    """Verify clear + add_situations mimics a fresh load."""

    def test_clear_then_load(self):
        mem = FinancialSituationMemory("test", max_entries=5)
        mem.add_situations([("old", "old_rec")])
        assert len(mem.documents) == 1

        mem.clear()
        assert len(mem.documents) == 0
        assert mem.bm25 is None

        mem.add_situations([("new", "new_rec")])
        assert len(mem.documents) == 1
        assert mem.documents[0] == "new"


class TestSQLiteToMemoryRoundtrip:
    """Simulate the full persistence roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        from tradingagents.learning.persistence import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "test.db"))

        # Simulate reflection saving memories
        situations = [
            ("bullish momentum with earnings beat", "increase position"),
            ("bearish divergence on RSI", "reduce exposure"),
        ]
        store.save_memories("bull_memory", situations, ticker="NVDA")

        # Simulate restart: load from SQLite into fresh BM25
        mem = FinancialSituationMemory("bull_memory", max_entries=100)
        loaded = store.load_memories("bull_memory")
        mem.add_situations(loaded)

        assert len(mem.documents) == 2

        # BM25 search works
        results = mem.get_memories("bullish earnings momentum", n_matches=1)
        assert len(results) == 1
        assert "bullish" in results[0]["matched_situation"]
        assert "increase position" in results[0]["recommendation"]

    def test_load_respects_max_entries(self, tmp_path):
        from tradingagents.learning.persistence import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "test.db"))

        # Save 10 memories
        situations = [(f"situation {i}", f"rec {i}") for i in range(10)]
        store.save_memories("test_mem", situations)

        # Load into memory with max_entries=5
        mem = FinancialSituationMemory("test_mem", max_entries=5)
        loaded = store.load_memories("test_mem")
        mem.add_situations(loaded)

        # Only newest 5 kept
        assert len(mem.documents) == 5
        assert mem.documents[-1] == "situation 9"
