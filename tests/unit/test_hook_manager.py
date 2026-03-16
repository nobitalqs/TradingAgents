"""Tests for HookManager: register, dispatch, unregister, error handling."""

from __future__ import annotations

import logging
from dataclasses import replace

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent
from tradingagents.hooks.hook_manager import HookManager


# ── Helpers ──────────────────────────────────────────────────────────


class _AppendHook(BaseHook):
    """Appends its name to metadata["order"] list on handle."""

    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def __init__(self, hook_name: str = "append", *, config: dict | None = None):
        super().__init__(config=config)
        self.name = hook_name

    def handle(self, context: HookContext) -> HookContext:
        order = list(context.metadata.get("order", []))
        order.append(self.name)
        return replace(context, metadata={**context.metadata, "order": order})


class _SkipHook(BaseHook):
    """Sets skip=True to halt the chain."""

    name = "skipper"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        return replace(context, skip=True)


class _CrashHook(BaseHook):
    """Always raises an exception."""

    name = "crasher"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        raise RuntimeError("boom")


# ── Tests ────────────────────────────────────────────────────────────


class TestHookManager:
    def test_register_and_dispatch(self):
        mgr = HookManager()
        mgr.register(_AppendHook("first"))
        mgr.register(_AppendHook("second"))

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
        result = mgr.dispatch(ctx)

        assert result.metadata["order"] == ["first", "second"]

    def test_skip_stops_chain(self):
        mgr = HookManager()
        mgr.register(_AppendHook("first"))
        mgr.register(_SkipHook())
        mgr.register(_AppendHook("third"))

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
        result = mgr.dispatch(ctx)

        # "first" ran, skipper set skip, "third" should NOT run
        assert result.skip is True
        assert "third" not in result.metadata.get("order", [])

    def test_failing_hook_does_not_crash(self, caplog):
        mgr = HookManager()
        mgr.register(_CrashHook())
        mgr.register(_AppendHook("after_crash"))

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
        with caplog.at_level(logging.ERROR):
            result = mgr.dispatch(ctx)

        # Chain continues despite the crash
        assert result.metadata["order"] == ["after_crash"]
        assert "boom" in caplog.text

    def test_unregister(self):
        mgr = HookManager()
        mgr.register(_AppendHook("keep"))
        mgr.register(_AppendHook("remove"))
        mgr.unregister("remove")

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
        result = mgr.dispatch(ctx)

        assert result.metadata["order"] == ["keep"]

    def test_unsubscribed_event_is_noop(self):
        mgr = HookManager()
        mgr.register(_AppendHook("only_propagate"))

        ctx = HookContext(event=HookEvent.AFTER_DECISION)
        result = mgr.dispatch(ctx)

        # No hooks subscribed to AFTER_DECISION -> context unchanged
        assert result.metadata.get("order") is None

    def test_summary(self):
        mgr = HookManager()
        mgr.register(_AppendHook("alpha"))
        mgr.register(_SkipHook())

        summary = mgr.summary
        assert summary["total"] == 2
        names = {h["name"] for h in summary["hooks"]}
        assert names == {"alpha", "skipper"}

    def test_dispatch_returns_original_when_no_hooks(self):
        mgr = HookManager()
        ctx = HookContext(event=HookEvent.HEARTBEAT_TICK, ticker="TEST")
        result = mgr.dispatch(ctx)
        assert result.ticker == "TEST"
