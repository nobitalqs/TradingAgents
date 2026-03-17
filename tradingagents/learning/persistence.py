"""SQLite-based persistence for TradingAgents learning layer."""

import hashlib
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    memory_name  TEXT NOT NULL,
    situation    TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'reflection',
    ticker       TEXT NOT NULL DEFAULT '',
    trade_date   TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    UNIQUE(memory_name, content_hash)
);

CREATE TABLE IF NOT EXISTS analysis_results (
    ticker          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    signal          TEXT NOT NULL,
    confidence      TEXT NOT NULL DEFAULT '',
    full_decision   TEXT NOT NULL DEFAULT '',
    state_json      TEXT NOT NULL DEFAULT '',
    reflected       INTEGER NOT NULL DEFAULT 0,
    actual_return   REAL,
    direction_correct INTEGER,
    UNIQUE(ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS run_timing (
    ticker          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL,
    created_at      TEXT NOT NULL
);
"""


class MemoryStore:
    """SQLite-backed store for memories, analysis results, and timing."""

    def __init__(self, db_path: str = "./tradingagents_memory.db") -> None:
        self.db_path = Path(db_path)
        self._ensure_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create the database file and tables if they don't exist."""
        needs_create = not self.db_path.exists()
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

        if needs_create:
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                logger.warning("Could not set file permissions on %s", self.db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for DB connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _content_hash(situation: str, recommendation: str) -> str:
        payload = f"{situation}|{recommendation}"
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def save_memories(
        self,
        memory_name: str,
        situations: list[tuple[str, str]],
        source: str = "reflection",
        ticker: str = "",
        trade_date: str = "",
    ) -> None:
        """Save (situation, recommendation) pairs. Idempotent via content_hash."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            for situation, recommendation in situations:
                content_hash = self._content_hash(situation, recommendation)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO memories
                        (memory_name, situation, recommendation, content_hash,
                         source, ticker, trade_date, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_name,
                        situation,
                        recommendation,
                        content_hash,
                        source,
                        ticker,
                        trade_date,
                        now,
                    ),
                )

    def load_memories(self, memory_name: str) -> list[tuple[str, str]]:
        """Load all (situation, recommendation) pairs for a memory.

        Returns in insertion order (oldest first) so that FIFO eviction
        in FinancialSituationMemory keeps the newest entries.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT situation, recommendation FROM memories "
                "WHERE memory_name = ? ORDER BY rowid",
                (memory_name,),
            ).fetchall()
        return [(row["situation"], row["recommendation"]) for row in rows]

    def get_memory_count(self, memory_name: str) -> int:
        """Count memories for a given memory name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE memory_name = ?",
                (memory_name,),
            ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Analysis results
    # ------------------------------------------------------------------

    def save_analysis_result(
        self,
        ticker: str,
        trade_date: str,
        signal: str,
        confidence: str,
        full_decision: str,
        state_json: str = "",
    ) -> None:
        """Upsert analysis result."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_results
                    (ticker, trade_date, signal, confidence, full_decision, state_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, trade_date) DO UPDATE SET
                    signal        = excluded.signal,
                    confidence    = excluded.confidence,
                    full_decision = excluded.full_decision,
                    state_json    = excluded.state_json
                """,
                (ticker, trade_date, signal, confidence, full_decision, state_json),
            )

    def get_pending_reflections(
        self, horizon_days: int = 7, as_of_date: str = ""
    ) -> list[dict]:
        """Get analysis results that need T+N reflection.

        Returns rows where ``reflected = 0`` and the trade_date is at least
        *horizon_days* before *as_of_date* (defaults to today).
        """
        if not as_of_date:
            as_of_date = datetime.utcnow().strftime("%Y-%m-%d")

        cutoff = (
            datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=horizon_days)
        ).strftime("%Y-%m-%d")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, trade_date, signal, confidence, full_decision,
                       state_json
                FROM analysis_results
                WHERE reflected = 0 AND trade_date <= ?
                """,
                (cutoff,),
            ).fetchall()

        return [dict(row) for row in rows]

    def update_reflection(
        self,
        ticker: str,
        trade_date: str,
        actual_return: float,
        direction_correct: bool,
    ) -> None:
        """Backfill reflection results."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE analysis_results
                SET reflected = 1,
                    actual_return = ?,
                    direction_correct = ?
                WHERE ticker = ? AND trade_date = ?
                """,
                (actual_return, int(direction_correct), ticker, trade_date),
            )

    def get_reflected_results(
        self, ticker: str = "", limit: int = 0
    ) -> list[dict]:
        """Get all reflected analysis results, optionally filtered by ticker.

        Returns rows where reflected=1, ordered by trade_date.
        """
        query = """
            SELECT ticker, trade_date, signal, confidence,
                   actual_return, direction_correct
            FROM analysis_results
            WHERE reflected = 1
        """
        params: list = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        query += " ORDER BY trade_date"
        if limit > 0:
            query += f" LIMIT {limit}"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def save_timing(self, ticker: str, trade_date: str, elapsed_seconds: float) -> None:
        """Record run timing."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_timing (ticker, trade_date, elapsed_seconds, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (ticker, trade_date, elapsed_seconds, now),
            )
