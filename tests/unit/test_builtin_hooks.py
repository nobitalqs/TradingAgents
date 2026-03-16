"""Tests for builtin hooks: JournalHook, RateLimitHook, DataIntegrityHook."""

from __future__ import annotations

import json
from pathlib import Path

from tradingagents.hooks.base import HookContext, HookEvent
from tradingagents.hooks.builtin.journal_hook import JournalHook
from tradingagents.hooks.builtin.ratelimit_hook import RateLimitHook
from tradingagents.hooks.builtin.integrity_hook import DataIntegrityHook


# ── JournalHook ──────────────────────────────────────────────────────


class TestJournalHook:
    def test_writes_jsonl_after_analyst(self, tmp_path: Path):
        hook = JournalHook(config={"output_dir": str(tmp_path)})
        ctx = HookContext(
            event=HookEvent.AFTER_ANALYST,
            ticker="AAPL",
            trade_date="2025-06-01",
            metadata={"analyst_type": "fundamental"},
        )
        hook.handle(ctx)

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "after_analyst"
        assert entry["ticker"] == "AAPL"
        assert entry["trade_date"] == "2025-06-01"
        assert entry["analyst_type"] == "fundamental"

    def test_writes_jsonl_after_debate(self, tmp_path: Path):
        hook = JournalHook(config={"output_dir": str(tmp_path)})
        ctx = HookContext(
            event=HookEvent.AFTER_DEBATE,
            ticker="GOOG",
            trade_date="2025-06-02",
            metadata={"debate_type": "bull_vs_bear"},
        )
        hook.handle(ctx)

        files = list(tmp_path.glob("*.jsonl"))
        entry = json.loads(files[0].read_text().strip())
        assert entry["debate_type"] == "bull_vs_bear"

    def test_writes_jsonl_after_decision(self, tmp_path: Path):
        hook = JournalHook(config={"output_dir": str(tmp_path)})
        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="TSLA",
            trade_date="2025-06-03",
            metadata={"decision": "buy", "confidence": 0.85},
        )
        hook.handle(ctx)

        files = list(tmp_path.glob("*.jsonl"))
        entry = json.loads(files[0].read_text().strip())
        assert entry["decision"] == "buy"
        assert entry["confidence"] == 0.85

    def test_appends_multiple_entries(self, tmp_path: Path):
        hook = JournalHook(config={"output_dir": str(tmp_path)})
        for i in range(3):
            ctx = HookContext(
                event=HookEvent.AFTER_ANALYST,
                ticker=f"T{i}",
                metadata={"analyst_type": "technical"},
            )
            hook.handle(ctx)

        files = list(tmp_path.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 3

    def test_subscriptions(self):
        hook = JournalHook()
        assert HookEvent.AFTER_ANALYST in hook.subscriptions
        assert HookEvent.AFTER_DEBATE in hook.subscriptions
        assert HookEvent.AFTER_DECISION in hook.subscriptions


# ── RateLimitHook ────────────────────────────────────────────────────


class TestRateLimitHook:
    def test_allows_under_limit(self):
        hook = RateLimitHook(config={"max_calls_per_second": 100, "window_seconds": 1})
        ctx = HookContext(
            event=HookEvent.BEFORE_TOOL_CALL,
            metadata={"tool_name": "get_stock_data"},
        )
        result = hook.handle(ctx)
        # Should not skip
        assert result.skip is False

    def test_tracks_stats_after_tool_call(self):
        hook = RateLimitHook(config={"max_calls_per_second": 100, "window_seconds": 1})
        ctx = HookContext(
            event=HookEvent.AFTER_TOOL_CALL,
            metadata={
                "tool_name": "get_stock_data",
                "duration_ms": 150.0,
                "error": None,
            },
        )
        hook.handle(ctx)
        summary = hook.stats_summary
        assert summary["total_calls"] == 1
        assert summary["total_errors"] == 0
        assert summary["total_ms"] == 150.0

    def test_tracks_errors(self):
        hook = RateLimitHook(config={"max_calls_per_second": 100, "window_seconds": 1})
        ctx = HookContext(
            event=HookEvent.AFTER_TOOL_CALL,
            metadata={
                "tool_name": "get_stock_data",
                "duration_ms": 50.0,
                "error": "timeout",
            },
        )
        hook.handle(ctx)
        assert hook.stats_summary["total_errors"] == 1

    def test_subscriptions(self):
        hook = RateLimitHook()
        assert HookEvent.BEFORE_TOOL_CALL in hook.subscriptions
        assert HookEvent.AFTER_TOOL_CALL in hook.subscriptions


# ── DataIntegrityHook ────────────────────────────────────────────────


class TestDataIntegrityHook:
    def test_flags_low_credibility(self):
        hook = DataIntegrityHook()
        ctx = HookContext(
            event=HookEvent.AFTER_ANALYST,
            ticker="NVDA",
            metadata={
                "data_credibility": {
                    "unreliable_count": 5,
                    "warnings": ["stale data", "missing fields"],
                }
            },
        )
        result = hook.handle(ctx)
        assert result.inject_context is not None
        assert "integrity" in result.inject_context.lower() or "warning" in result.inject_context.lower()

    def test_no_flag_when_credibility_ok(self):
        hook = DataIntegrityHook()
        ctx = HookContext(
            event=HookEvent.AFTER_ANALYST,
            ticker="NVDA",
            metadata={
                "data_credibility": {
                    "unreliable_count": 0,
                    "warnings": [],
                }
            },
        )
        result = hook.handle(ctx)
        assert result.inject_context is None

    def test_no_flag_when_no_credibility_metadata(self):
        hook = DataIntegrityHook()
        ctx = HookContext(
            event=HookEvent.AFTER_ANALYST,
            ticker="NVDA",
            metadata={},
        )
        result = hook.handle(ctx)
        assert result.inject_context is None

    def test_subscriptions(self):
        hook = DataIntegrityHook()
        assert HookEvent.AFTER_ANALYST in hook.subscriptions
