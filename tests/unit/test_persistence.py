"""Tests for tradingagents.learning.persistence.MemoryStore."""

import sqlite3
from pathlib import Path

import pytest

from tradingagents.learning.persistence import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """Return a MemoryStore backed by a temporary database."""
    db_path = tmp_path / "test_memory.db"
    return MemoryStore(db_path=str(db_path))


# ------------------------------------------------------------------
# Database creation
# ------------------------------------------------------------------


def test_create_db(tmp_path: Path) -> None:
    """DB file is created and contains the expected tables."""
    db_path = tmp_path / "new.db"
    MemoryStore(db_path=str(db_path))

    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = sorted(row[0] for row in cursor.fetchall())
    conn.close()

    assert "analysis_results" in tables
    assert "memories" in tables
    assert "run_timing" in tables


# ------------------------------------------------------------------
# Memories
# ------------------------------------------------------------------


def test_save_and_load_memories(store: MemoryStore) -> None:
    """Round-trip save then load returns the same pairs."""
    pairs = [
        ("market crash", "sell equities"),
        ("low volatility", "hold position"),
    ]
    store.save_memories("risk_lessons", pairs, source="backtest")
    loaded = store.load_memories("risk_lessons")

    assert loaded == pairs


def test_idempotent_save(store: MemoryStore) -> None:
    """Saving the same content twice does not create duplicates."""
    pairs = [("bull run", "increase exposure")]
    store.save_memories("strategy", pairs)
    store.save_memories("strategy", pairs)  # duplicate

    loaded = store.load_memories("strategy")
    assert len(loaded) == 1


def test_memory_count(store: MemoryStore) -> None:
    """Count starts at 0 and increments after saves."""
    assert store.get_memory_count("empty") == 0

    store.save_memories("signals", [("a", "b"), ("c", "d")])
    assert store.get_memory_count("signals") == 2


# ------------------------------------------------------------------
# Analysis results
# ------------------------------------------------------------------


def test_save_analysis_result(store: MemoryStore) -> None:
    """Insert and upsert analysis results."""
    store.save_analysis_result(
        ticker="AAPL",
        trade_date="2025-01-10",
        signal="BUY",
        confidence="HIGH",
        full_decision="strong buy based on momentum",
    )

    # Upsert with updated signal
    store.save_analysis_result(
        ticker="AAPL",
        trade_date="2025-01-10",
        signal="HOLD",
        confidence="MEDIUM",
        full_decision="revised to hold after news",
    )

    # Should have exactly one row (upsert, not duplicate)
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_results WHERE ticker = ? AND trade_date = ?",
            ("AAPL", "2025-01-10"),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["signal"] == "HOLD"
    assert rows[0]["confidence"] == "MEDIUM"


# ------------------------------------------------------------------
# Reflections
# ------------------------------------------------------------------


def test_get_pending_reflections(store: MemoryStore) -> None:
    """Only unreflected results older than horizon are returned."""
    # Old enough (trade_date + 7 days <= as_of_date)
    store.save_analysis_result(
        ticker="AAPL",
        trade_date="2025-01-01",
        signal="BUY",
        confidence="HIGH",
        full_decision="buy",
    )
    # Too recent (trade_date + 7 days > as_of_date)
    store.save_analysis_result(
        ticker="GOOG",
        trade_date="2025-01-09",
        signal="SELL",
        confidence="LOW",
        full_decision="sell",
    )

    pending = store.get_pending_reflections(horizon_days=7, as_of_date="2025-01-10")

    tickers = [r["ticker"] for r in pending]
    assert "AAPL" in tickers
    assert "GOOG" not in tickers


def test_update_reflection(store: MemoryStore) -> None:
    """update_reflection sets reflected=1 and stores actual_return."""
    store.save_analysis_result(
        ticker="MSFT",
        trade_date="2025-02-01",
        signal="BUY",
        confidence="HIGH",
        full_decision="buy msft",
    )

    store.update_reflection(
        ticker="MSFT",
        trade_date="2025-02-01",
        actual_return=0.05,
        direction_correct=True,
    )

    with store._connect() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_results WHERE ticker = ? AND trade_date = ?",
            ("MSFT", "2025-02-01"),
        ).fetchone()

    assert row["reflected"] == 1
    assert row["actual_return"] == pytest.approx(0.05)
    assert row["direction_correct"] == 1

    # Should no longer appear in pending reflections
    pending = store.get_pending_reflections(horizon_days=7, as_of_date="2025-03-01")
    assert all(r["ticker"] != "MSFT" for r in pending)


# ------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------


def test_save_timing(store: MemoryStore) -> None:
    """Inserts a timing record that can be read back."""
    store.save_timing(ticker="TSLA", trade_date="2025-03-01", elapsed_seconds=12.5)

    with store._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM run_timing WHERE ticker = ? AND trade_date = ?",
            ("TSLA", "2025-03-01"),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["elapsed_seconds"] == pytest.approx(12.5)
