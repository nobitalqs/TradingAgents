"""T+N automatic reflection — fetches actual stock prices and evaluates decisions.

After a trading decision is made, AutoReflector waits N days, then:
1. Fetches actual stock prices from yfinance (T+0 and T+N)
2. Computes price change % and direction correctness
3. Calls reflect_memories() to generate LLM reflections
4. Persists reflections to MemoryStore and marks the record as reflected
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from tradingagents.graph.reflection import Reflector, reflect_memories
from tradingagents.learning.persistence import MemoryStore

logger = logging.getLogger(__name__)


class AutoReflector:
    """Orchestrates delayed reflection on trading decisions.

    Parameters
    ----------
    reflector : Reflector
        LLM-backed reflector instance (calls all 5 component reflectors).
    memory_store : MemoryStore
        SQLite persistence for analysis results and memories.
    memories : dict[str, Any]
        Dict with keys: bull_memory, bear_memory, trader_memory,
        invest_judge_memory, risk_manager_memory.
    horizon : int
        Number of calendar days to wait before reflecting (default 7).
    """

    def __init__(
        self,
        reflector: Reflector,
        memory_store: MemoryStore,
        memories: dict[str, Any],
        horizon: int = 7,
    ) -> None:
        self._reflector = reflector
        self._store = memory_store
        self._memories = memories
        self._horizon = horizon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(
        self,
        ticker: str,
        trade_date: str,
        signal: str,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Reflect on a single past decision using actual market data.

        Returns a result dict on success, or None if prices unavailable.
        """
        prices = self._fetch_prices(ticker, trade_date, self._horizon)
        if prices is None:
            logger.warning(
                "Cannot reflect on %s %s: price data unavailable",
                ticker,
                trade_date,
            )
            return None

        price_t0, price_tn = prices
        price_change_pct = ((price_tn - price_t0) / price_t0) * 100
        direction_correct = self._check_direction(signal, price_change_pct)

        # Build human-readable returns description for the LLM
        returns_desc = self._describe_returns(
            ticker, signal, price_t0, price_tn, price_change_pct, direction_correct
        )

        # Run 5-component reflection (LLM calls)
        reflect_memories(self._reflector, state, returns_desc, self._memories)

        # Persist new memories to SQLite
        self._persist_memories(ticker, trade_date)

        # Mark as reflected in DB
        self._store.update_reflection(
            ticker=ticker,
            trade_date=trade_date,
            actual_return=round(price_change_pct, 4),
            direction_correct=direction_correct if direction_correct is not None else False,
        )

        result = {
            "ticker": ticker,
            "trade_date": trade_date,
            "signal": signal,
            "price_t0": price_t0,
            "price_tn": price_tn,
            "price_change_pct": round(price_change_pct, 4),
            "direction_correct": direction_correct,
        }
        logger.info(
            "Reflection complete: %s %s signal=%s change=%.2f%% correct=%s",
            ticker,
            trade_date,
            signal,
            price_change_pct,
            direction_correct,
        )
        return result

    def reflect_pending(self, as_of_date: str = "") -> list[dict[str, Any]]:
        """Process all pending reflections that have reached the horizon.

        Returns list of reflection results (one per record processed).
        """
        pending = self._store.get_pending_reflections(
            horizon_days=self._horizon, as_of_date=as_of_date
        )
        if not pending:
            logger.info("No pending reflections found")
            return []

        logger.info("Found %d pending reflections", len(pending))
        results: list[dict[str, Any]] = []

        for record in pending:
            ticker = record["ticker"]
            trade_date = record["trade_date"]
            signal = record["signal"]

            # Reconstruct state from stored JSON
            state = self._load_state(record.get("state_json", ""))
            if not state:
                logger.warning(
                    "Skipping %s %s: no state_json stored", ticker, trade_date
                )
                continue

            result = self.reflect(ticker, trade_date, signal, state)
            if result is not None:
                results.append(result)

        logger.info(
            "Reflected %d of %d pending records", len(results), len(pending)
        )
        return results

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def _fetch_prices(
        self, ticker: str, trade_date: str, horizon: int
    ) -> tuple[float, float] | None:
        """Fetch closing prices at T+0 and T+N from yfinance.

        Uses a ±3 day buffer to handle weekends and holidays.
        Returns (price_t0, price_tn) or None on failure.
        """
        try:
            t0 = datetime.strptime(trade_date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid trade_date format: %s", trade_date)
            return None

        tn = t0 + timedelta(days=horizon)
        # Fetch with buffer for non-trading days
        start = (t0 - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (tn + timedelta(days=5)).strftime("%Y-%m-%d")

        for attempt in range(2):
            try:
                data = yf.download(
                    ticker, start=start, end=end, progress=False, auto_adjust=True
                )
                if data is None or data.empty:
                    if attempt == 0:
                        time.sleep(1)
                        continue
                    return None

                price_t0 = self._find_closest_price(data, t0)
                price_tn = self._find_closest_price(data, tn)

                if price_t0 is None or price_tn is None:
                    return None

                return (price_t0, price_tn)

            except Exception:
                logger.exception(
                    "Failed to fetch prices for %s (attempt %d)", ticker, attempt + 1
                )
                if attempt == 0:
                    time.sleep(1)

        return None

    @staticmethod
    def _find_closest_price(
        data: Any, target: datetime, max_days: int = 3
    ) -> float | None:
        """Find the closing price closest to target within ±max_days."""
        for offset in range(max_days + 1):
            for delta in (timedelta(days=offset), timedelta(days=-offset)):
                check = target + delta
                check_str = check.strftime("%Y-%m-%d")
                try:
                    if check_str in data.index.strftime("%Y-%m-%d"):
                        mask = data.index.strftime("%Y-%m-%d") == check_str
                        close = data.loc[mask, "Close"]
                        if not close.empty:
                            val = close.iloc[0]
                            # Handle MultiIndex columns from yf.download
                            if hasattr(val, "iloc"):
                                val = val.iloc[0]
                            return float(val)
                except (KeyError, IndexError):
                    continue
        return None

    # ------------------------------------------------------------------
    # Direction checking
    # ------------------------------------------------------------------

    @staticmethod
    def _check_direction(
        signal: str, price_change_pct: float
    ) -> bool | None:
        """Check if the trading signal was directionally correct.

        - BUY correct if price went up (positive change)
        - SELL correct if price went down (negative change)
        - HOLD → None (not directional)
        """
        signal_upper = signal.upper().strip()
        if signal_upper == "BUY":
            return price_change_pct > 0
        if signal_upper == "SELL":
            return price_change_pct < 0
        return None  # HOLD or unknown

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_returns(
        ticker: str,
        signal: str,
        price_t0: float,
        price_tn: float,
        price_change_pct: float,
        direction_correct: bool | None,
    ) -> str:
        """Build a human-readable returns description for the LLM reflector."""
        direction = "UP" if price_change_pct > 0 else "DOWN"
        correctness = (
            "CORRECT"
            if direction_correct is True
            else "INCORRECT"
            if direction_correct is False
            else "NEUTRAL (HOLD)"
        )

        return (
            f"Stock: {ticker}\n"
            f"Signal: {signal}\n"
            f"Price at decision (T+0): ${price_t0:.2f}\n"
            f"Price after horizon (T+N): ${price_tn:.2f}\n"
            f"Price change: {price_change_pct:+.2f}% ({direction})\n"
            f"Direction assessment: {correctness}\n"
        )

    @staticmethod
    def _load_state(state_json: str) -> dict[str, Any] | None:
        """Deserialize state_json from DB. Returns None on failure."""
        if not state_json:
            return None
        try:
            return json.loads(state_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse state_json")
            return None

    def _persist_memories(self, ticker: str, trade_date: str) -> None:
        """Save current BM25 memory contents to SQLite."""
        for name, memory in self._memories.items():
            docs = getattr(memory, "documents", [])
            recs = getattr(memory, "recommendations", [])
            if not docs:
                continue

            situations = list(zip(docs, recs))
            self._store.save_memories(
                memory_name=name,
                situations=situations,
                source="reflection",
                ticker=ticker,
                trade_date=trade_date,
            )
