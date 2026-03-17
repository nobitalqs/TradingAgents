"""Integration tests: MarketHeartbeat + HookManager alert pipeline."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from tradingagents.hooks.base import HookContext, HookEvent
from tradingagents.hooks.hook_manager import HookManager
from tradingagents.orchestrator.heartbeat import MarketHeartbeat


def _make_history(close: float, volume: float) -> pd.DataFrame:
    """Build a minimal 5-row DataFrame mimicking yfinance output."""
    return pd.DataFrame(
        {"Close": [close] * 5, "Volume": [volume] * 5}
    )


@pytest.fixture
def hook_manager():
    return HookManager()


@pytest.fixture
def heartbeat_config():
    return {
        "heartbeat": {
            "interval_seconds": 0,  # no real sleep in tests
            "watchlist": ["NVDA"],
            "price_change_threshold": 0.03,
            "volume_spike_ratio": 2.0,
        }
    }


class TestHeartbeatAlertGeneration:
    """_tick detects anomalies and dispatches hooks."""

    @pytest.mark.asyncio
    async def test_price_spike_generates_alert(self, hook_manager, heartbeat_config):
        """A price change ≥ threshold triggers a price_spike alert."""
        dispatched: list[HookContext] = []
        original_dispatch = hook_manager.dispatch

        def tracking_dispatch(ctx):
            dispatched.append(ctx)
            return original_dispatch(ctx)

        hook_manager.dispatch = tracking_dispatch

        hb = MarketHeartbeat(heartbeat_config, hook_manager)

        # Manually set baselines (skip real yfinance)
        hb._baseline_prices["NVDA"] = 100.0
        hb._baseline_volumes["NVDA"] = 1_000_000.0

        # Patch _fetch_5d_history to return a 5% price spike
        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(105.0, 1_000_000.0),
        ):
            await hb._tick()

        # Check pending events
        events = hb.pending_events
        assert len(events) >= 1
        price_alerts = [e for e in events if e["type"] == "price_spike"]
        assert len(price_alerts) == 1
        assert price_alerts[0]["ticker"] == "NVDA"
        assert price_alerts[0]["change_pct"] >= 0.03

        # Check hook dispatch
        alert_hooks = [
            c for c in dispatched if c.event == HookEvent.HEARTBEAT_ALERT
        ]
        assert len(alert_hooks) >= 1

    @pytest.mark.asyncio
    async def test_volume_spike_generates_alert(self, hook_manager, heartbeat_config):
        hb = MarketHeartbeat(heartbeat_config, hook_manager)
        hb._baseline_prices["NVDA"] = 100.0
        hb._baseline_volumes["NVDA"] = 1_000_000.0

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(100.0, 3_000_000.0),
        ):
            await hb._tick()

        events = hb.pending_events
        volume_alerts = [e for e in events if e["type"] == "volume_spike"]
        assert len(volume_alerts) == 1
        assert volume_alerts[0]["ratio"] >= 2.0

    @pytest.mark.asyncio
    async def test_no_alert_within_threshold(self, hook_manager, heartbeat_config):
        """Small price change → no alert."""
        hb = MarketHeartbeat(heartbeat_config, hook_manager)
        hb._baseline_prices["NVDA"] = 100.0
        hb._baseline_volumes["NVDA"] = 1_000_000.0

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(101.0, 1_100_000.0),
        ):
            await hb._tick()

        events = hb.pending_events
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_tick_dispatches_heartbeat_tick_event(
        self, hook_manager, heartbeat_config
    ):
        """Every tick dispatches HEARTBEAT_TICK regardless of alerts."""
        tick_events: list[HookContext] = []
        original = hook_manager.dispatch

        def track(ctx):
            if ctx.event == HookEvent.HEARTBEAT_TICK:
                tick_events.append(ctx)
            return original(ctx)

        hook_manager.dispatch = track

        hb = MarketHeartbeat(heartbeat_config, hook_manager)
        hb._baseline_prices["NVDA"] = 100.0
        hb._baseline_volumes["NVDA"] = 1_000_000.0

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(100.0, 1_000_000.0),
        ):
            await hb._tick()

        assert len(tick_events) == 1
        assert tick_events[0].metadata["watchlist"] == ["NVDA"]


class TestHeartbeatCooldown:
    """Cooldown prevents duplicate alerts for the same ticker."""

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_second_alert(
        self, hook_manager, heartbeat_config
    ):
        hb = MarketHeartbeat(heartbeat_config, hook_manager)
        hb._baseline_prices["NVDA"] = 100.0
        hb._baseline_volumes["NVDA"] = 1_000_000.0

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(110.0, 1_000_000.0),
        ):
            await hb._tick()
            first_events = hb.pending_events

            # Second tick within cooldown
            await hb._tick()
            second_events = hb.pending_events

        assert len(first_events) >= 1
        assert len(second_events) == 0  # suppressed by cooldown


class TestHeartbeatCallback:
    """on_alert async callback fires when alerts are generated."""

    @pytest.mark.asyncio
    async def test_on_alert_callback_invoked(self, hook_manager, heartbeat_config):
        alerts_received: list[dict] = []

        async def on_alert(alert: dict) -> None:
            alerts_received.append(alert)

        hb = MarketHeartbeat(heartbeat_config, hook_manager, on_alert=on_alert)
        hb._baseline_prices["NVDA"] = 100.0
        hb._baseline_volumes["NVDA"] = 1_000_000.0

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(110.0, 1_000_000.0),
        ):
            await hb._tick()

        assert len(alerts_received) >= 1
        assert alerts_received[0]["ticker"] == "NVDA"

    @pytest.mark.asyncio
    async def test_failing_callback_does_not_crash(
        self, hook_manager, heartbeat_config
    ):
        """A failing on_alert callback doesn't break the tick loop."""

        async def bad_callback(alert: dict) -> None:
            raise ValueError("callback broken")

        hb = MarketHeartbeat(heartbeat_config, hook_manager, on_alert=bad_callback)
        hb._baseline_prices["NVDA"] = 100.0
        hb._baseline_volumes["NVDA"] = 1_000_000.0

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(110.0, 1_000_000.0),
        ):
            # Should not raise despite bad callback
            await hb._tick()

        events = hb.pending_events
        assert len(events) >= 1  # alert still recorded


class TestHeartbeatBaselineInit:
    """_init_baselines fetches history and sets baseline prices/volumes."""

    @pytest.mark.asyncio
    async def test_init_baselines(self, hook_manager, heartbeat_config):
        hb = MarketHeartbeat(heartbeat_config, hook_manager)

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=_make_history(150.0, 2_000_000.0),
        ):
            await hb._init_baselines()

        assert hb._baseline_prices["NVDA"] == 150.0
        assert hb._baseline_volumes["NVDA"] == 2_000_000.0

    @pytest.mark.asyncio
    async def test_init_baselines_handles_empty_history(
        self, hook_manager, heartbeat_config
    ):
        hb = MarketHeartbeat(heartbeat_config, hook_manager)

        with patch(
            "tradingagents.orchestrator.heartbeat._fetch_5d_history",
            return_value=pd.DataFrame(),
        ):
            await hb._init_baselines()

        assert "NVDA" not in hb._baseline_prices
