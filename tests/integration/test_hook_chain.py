"""Integration tests: HookManager + multiple hooks chaining together."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent
from tradingagents.hooks.hook_manager import HookManager


# ── Test hooks ───────────────────────────────────────────────────────


class RecorderHook(BaseHook):
    """Records every event it receives."""

    name = "recorder"
    subscriptions = [HookEvent.BEFORE_PROPAGATE, HookEvent.AFTER_DECISION]

    def __init__(self, config=None):
        super().__init__(config=config)
        self.calls: list[HookContext] = []

    def handle(self, context: HookContext) -> HookContext:
        self.calls.append(context)
        return context


class InjectHook(BaseHook):
    """Injects text into context — downstream hooks should see it."""

    name = "inject"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        return replace(context, inject_context="portfolio: 100 shares NVDA")


class SkipHook(BaseHook):
    """Sets skip=True, halting the chain."""

    name = "skipper"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        return replace(context, skip=True)


class ExplodingHook(BaseHook):
    """Always raises — tests error isolation."""

    name = "exploder"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        raise RuntimeError("boom")


# ── Tests ────────────────────────────────────────────────────────────


class TestHookChainPropagation:
    """Verify context flows through a chain of hooks in order."""

    def test_inject_then_record(self):
        """InjectHook sets inject_context, RecorderHook sees it downstream."""
        mgr = HookManager()
        inject = InjectHook()
        recorder = RecorderHook()
        mgr.register(inject)
        mgr.register(recorder)

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="NVDA")
        result = mgr.dispatch(ctx)

        assert result.inject_context == "portfolio: 100 shares NVDA"
        assert len(recorder.calls) == 1
        assert recorder.calls[0].inject_context == "portfolio: 100 shares NVDA"

    def test_error_isolation_continues_chain(self):
        """A failing hook doesn't block the rest of the chain."""
        mgr = HookManager()
        exploder = ExplodingHook()
        recorder = RecorderHook()
        mgr.register(exploder)
        mgr.register(recorder)

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="TSLA")
        result = mgr.dispatch(ctx)

        # Recorder still fires after exploder fails
        assert len(recorder.calls) == 1
        assert result.ticker == "TSLA"
        assert not result.skip

    def test_skip_halts_chain(self):
        """skip=True prevents subsequent hooks from running."""
        mgr = HookManager()
        skipper = SkipHook()
        recorder = RecorderHook()
        mgr.register(skipper)
        mgr.register(recorder)

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="AAPL")
        result = mgr.dispatch(ctx)

        assert result.skip is True
        assert len(recorder.calls) == 0

    def test_events_are_isolated(self):
        """Dispatching one event does not trigger hooks for a different event."""
        mgr = HookManager()
        recorder = RecorderHook()
        mgr.register(recorder)

        ctx = HookContext(event=HookEvent.BEFORE_ANALYST, ticker="GOOG")
        mgr.dispatch(ctx)

        assert len(recorder.calls) == 0

    def test_unregister_removes_from_chain(self):
        """Unregistered hooks no longer fire."""
        mgr = HookManager()
        recorder = RecorderHook()
        mgr.register(recorder)
        mgr.unregister("recorder")

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="MSFT")
        mgr.dispatch(ctx)

        assert len(recorder.calls) == 0
        assert mgr.summary["total"] == 0


class TestBuiltinHookLoading:
    """Verify load_builtin_hooks wires up real hook classes from config."""

    def test_load_journal_hook(self, tmp_path):
        config = {
            "hooks": {
                "entries": {
                    "journal": {"enabled": True, "output_dir": str(tmp_path / "journal")},
                }
            }
        }
        mgr = HookManager(config=config)
        mgr.load_builtin_hooks()

        assert mgr.summary["total"] == 1
        assert mgr.summary["hooks"][0]["name"] == "journal"

        # Dispatch an event and check the JSONL file
        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="NVDA",
            trade_date="2026-01-15",
            metadata={"decision": "BUY", "confidence": "HIGH"},
        )
        mgr.dispatch(ctx)

        journal_file = tmp_path / "journal" / "journal.jsonl"
        assert journal_file.exists()
        entry = json.loads(journal_file.read_text().strip())
        assert entry["ticker"] == "NVDA"
        assert entry["decision"] == "BUY"

    def test_load_portfolio_hook_injects_context(self, tmp_path):
        portfolio = {
            "holdings": [
                {"symbol": "NVDA", "quantity": 50, "avg_cost": 120.0},
                {"symbol": "AAPL", "quantity": 100, "avg_cost": 150.0},
            ],
            "total_value": 21000.0,
        }
        pf_file = tmp_path / "portfolio.json"
        pf_file.write_text(json.dumps(portfolio))

        config = {
            "hooks": {
                "entries": {
                    "portfolio_context": {
                        "enabled": True,
                        "portfolio_file": str(pf_file),
                    },
                }
            }
        }
        mgr = HookManager(config=config)
        mgr.load_builtin_hooks()

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="NVDA")
        result = mgr.dispatch(ctx)

        assert result.inject_context is not None
        assert "NVDA" in result.inject_context
        assert "50 shares" in result.inject_context
        assert "$21,000.00" in result.inject_context

    def test_load_integrity_hook_flags_low_quality(self):
        config = {
            "hooks": {
                "entries": {
                    "data_integrity": {"enabled": True},
                }
            }
        }
        mgr = HookManager(config=config)
        mgr.load_builtin_hooks()

        ctx = HookContext(
            event=HookEvent.AFTER_ANALYST,
            metadata={
                "data_credibility": {
                    "unreliable_count": 3,
                    "warnings": ["Low price confidence"],
                }
            },
        )
        result = mgr.dispatch(ctx)

        assert result.inject_context is not None
        assert "DATA INTEGRITY WARNING" in result.inject_context

    def test_disabled_hooks_not_loaded(self):
        config = {
            "hooks": {
                "entries": {
                    "journal": {"enabled": False},
                    "ratelimit": {"enabled": False},
                }
            }
        }
        mgr = HookManager(config=config)
        mgr.load_builtin_hooks()

        assert mgr.summary["total"] == 0

    def test_multiple_builtins_compose(self, tmp_path):
        """Multiple builtins loaded together dispatch correctly on different events."""
        portfolio = {"holdings": [{"symbol": "TSLA", "quantity": 10, "avg_cost": 200.0}]}
        pf_file = tmp_path / "portfolio.json"
        pf_file.write_text(json.dumps(portfolio))

        config = {
            "hooks": {
                "entries": {
                    "journal": {"enabled": True, "output_dir": str(tmp_path / "j")},
                    "portfolio_context": {
                        "enabled": True,
                        "portfolio_file": str(pf_file),
                    },
                    "data_integrity": {"enabled": True},
                }
            }
        }
        mgr = HookManager(config=config)
        mgr.load_builtin_hooks()

        assert mgr.summary["total"] == 3

        # BEFORE_PROPAGATE → portfolio hook fires
        ctx1 = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="TSLA")
        r1 = mgr.dispatch(ctx1)
        assert "TSLA" in (r1.inject_context or "")

        # AFTER_DECISION → journal hook fires
        ctx2 = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="TSLA",
            metadata={"decision": "SELL"},
        )
        mgr.dispatch(ctx2)
        journal_file = tmp_path / "j" / "journal.jsonl"
        assert journal_file.exists()
