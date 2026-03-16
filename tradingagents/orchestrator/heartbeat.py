"""Real-time market heartbeat monitor for price and volume anomalies."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Awaitable

import yfinance as yf

from tradingagents.hooks.base import HookContext, HookEvent

logger = logging.getLogger(__name__)

# Minimum seconds between repeated alerts for the same ticker.
_DEFAULT_COOLDOWN_SECONDS = 15 * 60


class MarketHeartbeat:
    """Periodically checks watchlist tickers for price/volume anomalies.

    Parameters
    ----------
    config : dict
        Full application config; heartbeat reads ``config["heartbeat"]``.
    hook_manager : object
        Object with a synchronous ``dispatch(ctx) -> HookContext`` method.
    on_alert : callable | None
        Optional async callback ``(alert_dict) -> None`` fired on each anomaly.
    """

    def __init__(
        self,
        config: dict[str, Any],
        hook_manager: Any,
        on_alert: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        hb_cfg: dict[str, Any] = config.get("heartbeat", {})

        self._hook_manager = hook_manager
        self._on_alert = on_alert

        self._interval_seconds: int = hb_cfg.get("interval_seconds", 60)
        self._watchlist: list[str] = list(hb_cfg.get("watchlist", []))
        self._price_threshold: float = hb_cfg.get("price_change_threshold", 0.03)
        self._volume_ratio: float = hb_cfg.get("volume_spike_ratio", 2.0)

        self._baseline_prices: dict[str, float] = {}
        self._baseline_volumes: dict[str, float] = {}
        self._cooldown: dict[str, float] = {}
        self._pending_events: list[dict[str, Any]] = []

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize baselines and begin the tick loop."""
        self._running = True
        await self._init_baselines()

        while self._running:
            try:
                await self._tick()
            except Exception:
                logger.exception("Heartbeat tick failed")
            await asyncio.sleep(self._interval_seconds)

    def stop(self) -> None:
        """Signal the tick loop to exit."""
        self._running = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_events(self) -> list[dict[str, Any]]:
        """Return and clear accumulated alert events."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    async def _init_baselines(self) -> None:
        """Fetch 5-day history for each ticker to establish baselines."""
        loop = asyncio.get_running_loop()

        for ticker in self._watchlist:
            try:
                hist = await loop.run_in_executor(
                    None, _fetch_5d_history, ticker
                )
                if hist is not None and not hist.empty:
                    self._baseline_prices[ticker] = float(
                        hist["Close"].iloc[-1]
                    )
                    self._baseline_volumes[ticker] = float(
                        hist["Volume"].mean()
                    )
                else:
                    logger.warning("No history returned for %s", ticker)
            except Exception:
                logger.exception("Failed to init baseline for %s", ticker)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """One heartbeat cycle: dispatch tick event, check all tickers."""
        self._hook_manager.dispatch(
            HookContext(
                event=HookEvent.HEARTBEAT_TICK,
                metadata={"watchlist": list(self._watchlist)},
            )
        )

        now = time.time()
        for ticker in self._watchlist:
            alerts = await self._check_ticker(ticker)
            for alert in alerts:
                # Enforce cooldown per ticker
                last_alert_time = self._cooldown.get(ticker, 0.0)
                if now - last_alert_time < _DEFAULT_COOLDOWN_SECONDS:
                    continue

                self._cooldown[ticker] = now
                self._pending_events.append(alert)

                self._hook_manager.dispatch(
                    HookContext(
                        event=HookEvent.HEARTBEAT_ALERT,
                        metadata=alert,
                    )
                )

                if self._on_alert is not None:
                    try:
                        await self._on_alert(alert)
                    except Exception:
                        logger.exception("on_alert callback failed for %s", ticker)

    async def _check_ticker(self, ticker: str) -> list[dict[str, Any]]:
        """Check a single ticker for price change or volume spike.

        Returns a list of alert dicts (may be empty).
        """
        alerts: list[dict[str, Any]] = []

        loop = asyncio.get_running_loop()
        try:
            hist = await loop.run_in_executor(None, _fetch_5d_history, ticker)
        except Exception:
            logger.exception("Failed to fetch data for %s", ticker)
            return alerts

        if hist is None or hist.empty:
            return alerts

        current_price = float(hist["Close"].iloc[-1])
        current_volume = float(hist["Volume"].iloc[-1])

        baseline_price = self._baseline_prices.get(ticker)
        baseline_volume = self._baseline_volumes.get(ticker)

        # Price change check
        if baseline_price and baseline_price > 0:
            pct_change = abs(current_price - baseline_price) / baseline_price
            if pct_change >= self._price_threshold:
                alerts.append(
                    {
                        "ticker": ticker,
                        "type": "price_spike",
                        "current": current_price,
                        "baseline": baseline_price,
                        "change_pct": round(pct_change, 4),
                        "timestamp": time.time(),
                    }
                )

        # Volume spike check
        if baseline_volume and baseline_volume > 0:
            volume_ratio = current_volume / baseline_volume
            if volume_ratio >= self._volume_ratio:
                alerts.append(
                    {
                        "ticker": ticker,
                        "type": "volume_spike",
                        "current": current_volume,
                        "baseline": baseline_volume,
                        "ratio": round(volume_ratio, 2),
                        "timestamp": time.time(),
                    }
                )

        # Update baselines with latest values
        self._baseline_prices[ticker] = current_price
        self._baseline_volumes[ticker] = (
            baseline_volume * 0.8 + current_volume * 0.2
            if baseline_volume
            else current_volume
        )

        return alerts


def _fetch_5d_history(ticker: str) -> Any:
    """Synchronous helper — call from an executor."""
    t = yf.Ticker(ticker)
    return t.history(period="5d")
