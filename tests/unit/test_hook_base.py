"""Tests for hook base classes: HookEvent, HookContext, BaseHook."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace

import pytest

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent


# ── HookEvent ────────────────────────────────────────────────────────


class TestHookEvent:
    """HookEvent enum string values and membership."""

    def test_event_string_values(self):
        assert HookEvent.BEFORE_PROPAGATE == "before_propagate"
        assert HookEvent.AFTER_PROPAGATE == "after_propagate"
        assert HookEvent.BEFORE_ANALYST == "before_analyst"
        assert HookEvent.AFTER_ANALYST == "after_analyst"
        assert HookEvent.BEFORE_DEBATE == "before_debate"
        assert HookEvent.AFTER_DEBATE == "after_debate"
        assert HookEvent.BEFORE_DECISION == "before_decision"
        assert HookEvent.AFTER_DECISION == "after_decision"
        assert HookEvent.BEFORE_TOOL_CALL == "before_tool_call"
        assert HookEvent.AFTER_TOOL_CALL == "after_tool_call"
        assert HookEvent.HEARTBEAT_TICK == "heartbeat_tick"
        assert HookEvent.HEARTBEAT_ALERT == "heartbeat_alert"
        assert HookEvent.CRON_JOB_START == "cron_job_start"
        assert HookEvent.CRON_JOB_END == "cron_job_end"
        assert HookEvent.BEFORE_REFLECT == "before_reflect"
        assert HookEvent.AFTER_REFLECT == "after_reflect"

    def test_total_event_count(self):
        assert len(HookEvent) == 16

    def test_event_is_string(self):
        for event in HookEvent:
            assert isinstance(event, str)
            assert event.value == event  # str enum identity


# ── HookContext ──────────────────────────────────────────────────────


class TestHookContext:
    """HookContext dataclass defaults and immutability patterns."""

    def test_defaults(self):
        ctx = HookContext(event=HookEvent.HEARTBEAT_TICK)
        assert ctx.event == HookEvent.HEARTBEAT_TICK
        assert isinstance(ctx.timestamp, datetime)
        assert ctx.timestamp.tzinfo == timezone.utc
        assert ctx.ticker == ""
        assert ctx.trade_date == ""
        assert ctx.state == {}
        assert ctx.metadata == {}
        assert ctx.config == {}
        assert ctx.inject_context is None
        assert ctx.skip is False

    def test_context_mutation_returns_new(self):
        """Using dataclasses.replace to produce a new context (immutable pattern)."""
        original = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="AAPL")
        updated = replace(original, ticker="GOOG", skip=True)

        assert original.ticker == "AAPL"
        assert original.skip is False
        assert updated.ticker == "GOOG"
        assert updated.skip is True

    def test_context_with_all_fields(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            timestamp=ts,
            ticker="MSFT",
            trade_date="2025-01-01",
            state={"price": 400.0},
            metadata={"source": "test"},
            config={"verbose": True},
            inject_context="extra info",
            skip=True,
        )
        assert ctx.timestamp == ts
        assert ctx.ticker == "MSFT"
        assert ctx.inject_context == "extra info"


# ── BaseHook ─────────────────────────────────────────────────────────


class _DummyHook(BaseHook):
    """Concrete implementation for testing."""

    name = "dummy"
    subscriptions = [HookEvent.BEFORE_PROPAGATE, HookEvent.AFTER_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        return replace(context, metadata={**context.metadata, "handled": True})


class TestBaseHook:
    """BaseHook abstract interface contract."""

    def test_subclass_handle(self):
        hook = _DummyHook()
        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
        result = hook.handle(ctx)
        assert result.metadata["handled"] is True

    def test_subclass_name_and_subscriptions(self):
        hook = _DummyHook()
        assert hook.name == "dummy"
        assert HookEvent.BEFORE_PROPAGATE in hook.subscriptions

    def test_repr(self):
        hook = _DummyHook()
        r = repr(hook)
        assert "dummy" in r
        assert "before_propagate" in r or "BEFORE_PROPAGATE" in r

    def test_config_defaults_to_empty_dict(self):
        hook = _DummyHook()
        assert hook.config == {}

    def test_config_from_constructor(self):
        hook = _DummyHook(config={"key": "val"})
        assert hook.config == {"key": "val"}

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseHook()  # type: ignore[abstract]
