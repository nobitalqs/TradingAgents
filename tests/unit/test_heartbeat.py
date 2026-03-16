"""Tests for tradingagents.orchestrator.heartbeat."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.orchestrator.heartbeat import MarketHeartbeat, _DEFAULT_COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    watchlist: list[str] | None = None,
    price_threshold: float = 0.03,
    volume_ratio: float = 2.0,
) -> dict:
    return {
        "heartbeat": {
            "interval_seconds": 5,
            "watchlist": watchlist or ["AAPL"],
            "price_change_threshold": price_threshold,
            "volume_spike_ratio": volume_ratio,
        }
    }


def _make_history(close: float, volume: float) -> pd.DataFrame:
    """Return a tiny DataFrame that looks like yfinance 5d history."""
    return pd.DataFrame(
        {
            "Close": [close],
            "Volume": [volume],
        }
    )


@pytest.fixture()
def mock_hooks() -> MagicMock:
    hook = MagicMock()
    hook.dispatch = MagicMock(side_effect=lambda ctx: ctx)
    return hook


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckTickerPriceSpike:
    """_check_ticker should detect price changes above the threshold."""

    @pytest.mark.asyncio
    async def test_check_ticker_price_spike(self, mock_hooks: MagicMock) -> None:
        hb = MarketHeartbeat(
            config=_make_config(price_threshold=0.03, volume_ratio=100.0),
            hook_manager=mock_hooks,
        )
        # Set baseline: price=100, volume=1M
        hb._baseline_prices["AAPL"] = 100.0
        hb._baseline_volumes["AAPL"] = 1_000_000.0

        # Current price jumped 5%
        fake_hist = _make_history(close=105.0, volume=1_000_000.0)

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=fake_hist,
        ):
            alerts = await hb._check_ticker("AAPL")

        assert len(alerts) == 1
        assert alerts[0]["type"] == "price_spike"
        assert alerts[0]["ticker"] == "AAPL"
        assert alerts[0]["change_pct"] == pytest.approx(0.05, abs=0.001)


class TestCheckTickerNoAlert:
    """No alerts when price and volume are within normal ranges."""

    @pytest.mark.asyncio
    async def test_check_ticker_no_alert(self, mock_hooks: MagicMock) -> None:
        hb = MarketHeartbeat(
            config=_make_config(price_threshold=0.03, volume_ratio=2.0),
            hook_manager=mock_hooks,
        )
        hb._baseline_prices["AAPL"] = 100.0
        hb._baseline_volumes["AAPL"] = 1_000_000.0

        # 1% price move, normal volume
        fake_hist = _make_history(close=101.0, volume=1_200_000.0)

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=fake_hist,
        ):
            alerts = await hb._check_ticker("AAPL")

        assert len(alerts) == 0


class TestCheckTickerVolumeSpike:
    """_check_ticker should detect volume spikes above the ratio threshold."""

    @pytest.mark.asyncio
    async def test_check_ticker_volume_spike(self, mock_hooks: MagicMock) -> None:
        hb = MarketHeartbeat(
            config=_make_config(price_threshold=0.50, volume_ratio=2.0),
            hook_manager=mock_hooks,
        )
        hb._baseline_prices["AAPL"] = 100.0
        hb._baseline_volumes["AAPL"] = 1_000_000.0

        # Volume is 3x normal, price flat
        fake_hist = _make_history(close=100.5, volume=3_000_000.0)

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=fake_hist,
        ):
            alerts = await hb._check_ticker("AAPL")

        assert len(alerts) == 1
        assert alerts[0]["type"] == "volume_spike"
        assert alerts[0]["ratio"] == pytest.approx(3.0, abs=0.1)


class TestCooldown:
    """Cooldown should prevent duplicate alerts for the same ticker."""

    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate_alerts(
        self, mock_hooks: MagicMock
    ) -> None:
        hb = MarketHeartbeat(
            config=_make_config(price_threshold=0.03, volume_ratio=100.0),
            hook_manager=mock_hooks,
        )
        hb._baseline_prices["AAPL"] = 100.0
        hb._baseline_volumes["AAPL"] = 1_000_000.0

        fake_hist = _make_history(close=110.0, volume=1_000_000.0)

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=fake_hist,
        ):
            # First tick: should produce an alert
            await hb._tick()
            first_events = list(hb._pending_events)
            assert len(first_events) == 1

            # Second tick immediately after: cooldown blocks it
            await hb._tick()
            # pending_events should still be 1 (no new ones appended)
            assert len(hb._pending_events) == 1


class TestPendingEventsCleared:
    """pending_events property should drain the internal queue."""

    @pytest.mark.asyncio
    async def test_pending_events_cleared_after_read(
        self, mock_hooks: MagicMock
    ) -> None:
        hb = MarketHeartbeat(
            config=_make_config(price_threshold=0.01, volume_ratio=100.0),
            hook_manager=mock_hooks,
        )
        hb._baseline_prices["AAPL"] = 100.0
        hb._baseline_volumes["AAPL"] = 1_000_000.0

        fake_hist = _make_history(close=110.0, volume=1_000_000.0)

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=fake_hist,
        ):
            await hb._tick()

        events = hb.pending_events
        assert len(events) >= 1

        # Second read should return empty
        assert len(hb.pending_events) == 0
