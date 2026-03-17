"""Integration tests: cross-cutting flows connecting hooks, verification, and signals."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradingagents.graph.analyst_signals import (
    compute_consensus,
    create_extract_signals_node,
    extract_all_signals,
)
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.hooks.base import BaseHook, HookContext, HookEvent
from tradingagents.hooks.hook_manager import HookManager
from tradingagents.verification.data_verifier import DataVerifier


class CredibilityInjectorHook(BaseHook):
    """Runs DataVerifier and injects credibility summary into context."""

    name = "credibility_injector"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def __init__(self, verifier: DataVerifier, price_data: dict, news_items: list):
        super().__init__()
        self._verifier = verifier
        self._price_data = price_data
        self._news_items = news_items

    def handle(self, context: HookContext) -> HookContext:
        summary = self._verifier.build_credibility_summary(
            self._price_data, self._news_items
        )
        return replace(
            context,
            inject_context=summary.to_prompt_text(),
            metadata={**context.metadata, "data_credibility": summary.to_dict()},
        )


class DecisionRecorderHook(BaseHook):
    """Captures the final decision from AFTER_DECISION events."""

    name = "decision_recorder"
    subscriptions = [HookEvent.AFTER_DECISION]

    def __init__(self):
        super().__init__()
        self.decisions: list[dict] = []

    def handle(self, context: HookContext) -> HookContext:
        self.decisions.append(
            {
                "ticker": context.ticker,
                "decision": context.metadata.get("decision"),
                "confidence": context.metadata.get("confidence"),
                "credibility": context.metadata.get("data_credibility"),
            }
        )
        return context


class TestVerificationToHookFlow:
    """DataVerifier output flows through hooks into the analysis context."""

    def test_credibility_summary_injected_into_context(self):
        verifier = DataVerifier()
        hook = CredibilityInjectorHook(
            verifier=verifier,
            price_data={"yfinance": 150.0, "alpha_vantage": 150.3},
            news_items=[
                {"headline": "Strong earnings beat", "source": "Reuters"},
                {"headline": "Guaranteed profit act now", "source": "spam.io"},
            ],
        )

        mgr = HookManager()
        mgr.register(hook)

        ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="NVDA")
        result = mgr.dispatch(ctx)

        assert result.inject_context is not None
        assert "DATA RELIABILITY ASSESSMENT" in result.inject_context
        assert "Reuters" in result.inject_context

        cred = result.metadata["data_credibility"]
        assert cred["price_confidence"] == 0.95
        assert len(cred["news_items"]) == 2


class TestFullAnalysisPipeline:
    """Simulate a lightweight analysis: reports → consensus → signal → hooks."""

    def test_reports_to_decision_with_hooks(self, tmp_path):
        # 1. Set up hooks
        mgr = HookManager()
        recorder = DecisionRecorderHook()
        mgr.register(recorder)

        # 2. Simulate analyst reports
        state = {
            "market_report": "DIRECTION: BUY\nBullish momentum with breakout.",
            "sentiment_report": "DIRECTION: BUY\nOverwhelming positive sentiment.",
            "news_report": "DIRECTION: HOLD\nMixed macro signals.",
            "fundamentals_report": "DIRECTION: BUY\nRevenue growth impressive.",
        }

        # 3. Extract signals → consensus
        node_fn = create_extract_signals_node()
        result = node_fn(state)
        consensus = result["analyst_consensus"]

        # 4. Process signal
        decision_text = (
            f"Consensus: {consensus['buy_count']} BUY, {consensus['sell_count']} SELL. "
            f"Recommendation: BUY with {consensus['confidence']} confidence."
        )
        processor = SignalProcessor(llm=None)
        signal = processor.process_signal(decision_text)

        # 5. Dispatch AFTER_DECISION hook
        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="NVDA",
            trade_date="2026-01-15",
            metadata={
                "decision": signal,
                "confidence": consensus["confidence"],
            },
        )
        mgr.dispatch(ctx)

        # 6. Verify
        assert signal == "BUY"
        assert len(recorder.decisions) == 1
        assert recorder.decisions[0]["decision"] == "BUY"
        assert recorder.decisions[0]["confidence"] == "MEDIUM"

    def test_verification_enriched_decision_flow(self):
        """Verification → hooks → signal extraction → decision recording."""
        mgr = HookManager()

        # Credibility hook
        verifier = DataVerifier()
        cred_hook = CredibilityInjectorHook(
            verifier=verifier,
            price_data={"src1": 200.0, "src2": 210.0},  # >2% deviation
            news_items=[
                {"headline": "Breaking: Secret tip insider", "source": "shady.biz"},
            ],
        )
        mgr.register(cred_hook)

        # Decision recorder
        recorder = DecisionRecorderHook()
        mgr.register(recorder)

        # BEFORE_PROPAGATE → credibility injected
        ctx1 = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="NVDA")
        result1 = mgr.dispatch(ctx1)

        assert result1.inject_context is not None
        cred = result1.metadata["data_credibility"]
        assert cred["price_confidence"] == 0.5
        assert len(cred["warnings"]) >= 1

        # AFTER_DECISION → recorder captures decision with credibility
        ctx2 = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="NVDA",
            metadata={
                "decision": "HOLD",
                "confidence": "LOW",
                "data_credibility": cred,
            },
        )
        mgr.dispatch(ctx2)

        assert len(recorder.decisions) == 1
        assert recorder.decisions[0]["credibility"]["price_confidence"] == 0.5


class TestJournalWithVerification:
    """JournalHook persists credibility metadata alongside decisions."""

    def test_journal_records_credibility(self, tmp_path):
        config = {
            "hooks": {
                "entries": {
                    "journal": {"enabled": True, "output_dir": str(tmp_path / "j")},
                }
            }
        }
        mgr = HookManager(config=config)
        mgr.load_builtin_hooks()

        # Simulate a decision with credibility data
        verifier = DataVerifier()
        summary = verifier.build_credibility_summary(
            price_data={"a": 100.0, "b": 100.5},
            news_items=[{"headline": "Good news", "source": "Bloomberg"}],
        )

        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="NVDA",
            trade_date="2026-01-15",
            metadata={
                "decision": "BUY",
                "confidence": "HIGH",
                "price_confidence": summary.price_confidence,
            },
        )
        mgr.dispatch(ctx)

        journal_file = tmp_path / "j" / "journal.jsonl"
        entry = json.loads(journal_file.read_text().strip())
        assert entry["decision"] == "BUY"
        assert entry["price_confidence"] == 0.95
