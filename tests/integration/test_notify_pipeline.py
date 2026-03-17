"""Integration tests: NotifyHook → real Notifier backends via HookManager."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.hooks.base import HookContext, HookEvent
from tradingagents.hooks.builtin.notify_hook import NotifyHook, _create_notifier
from tradingagents.hooks.hook_manager import HookManager
from tradingagents.notify.feishu_notifier import FeishuNotifier
from tradingagents.notify.slack_notifier import SlackNotifier
from tradingagents.notify.webhook_notifier import WebhookNotifier


class TestNotifierFactory:
    """_create_notifier returns correct backend instances."""

    def test_create_feishu(self):
        notifier = _create_notifier("feishu", {"webhook_url": "https://x"})
        assert isinstance(notifier, FeishuNotifier)

    def test_create_slack(self):
        notifier = _create_notifier("slack", {"webhook_url": "https://x"})
        assert isinstance(notifier, SlackNotifier)

    def test_create_webhook(self):
        notifier = _create_notifier("webhook", {"url": "https://x"})
        assert isinstance(notifier, WebhookNotifier)

    def test_create_unknown_returns_none(self):
        assert _create_notifier("telegram", {}) is None


class TestNotifyHookFormatting:
    """Message formatting for different event types."""

    def test_after_decision_format(self):
        hook = NotifyHook(config={"notifier": "log"})
        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="NVDA",
            trade_date="2026-01-15",
            metadata={"decision": "BUY", "confidence": "HIGH", "full_signal": "Buy strongly."},
        )
        msg = hook._format_message(ctx)
        assert "NVDA" in msg
        assert "BUY" in msg
        assert "HIGH" in msg
        assert "Buy strongly." in msg

    def test_heartbeat_alert_format(self):
        hook = NotifyHook(config={"notifier": "log"})
        ctx = HookContext(
            event=HookEvent.HEARTBEAT_ALERT,
            metadata={"type": "price_spike", "ticker": "TSLA", "change_pct": 0.05},
        )
        msg = hook._format_message(ctx)
        assert "TSLA" in msg
        assert "price_spike" in msg

    def test_irrelevant_event_returns_empty(self):
        hook = NotifyHook(config={"notifier": "log"})
        ctx = HookContext(event=HookEvent.BEFORE_ANALYST)
        assert hook._format_message(ctx) == ""


class TestNotifyHookFeishuIntegration:
    """NotifyHook → FeishuNotifier end-to-end (mocked HTTP)."""

    @patch("tradingagents.notify.feishu_notifier.requests.post")
    def test_feishu_receives_decision_card(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "msg": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        hook = NotifyHook(config={
            "notifier": "feishu",
            "notifier_config": {
                "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/test",
            },
        })

        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="NVDA",
            trade_date="2026-01-15",
            metadata={"decision": "BUY", "confidence": "HIGH"},
        )

        # handle() spawns a daemon thread — call _send directly for deterministic test
        msg = hook._format_message(ctx)
        hook._send(msg)

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json", mock_post.call_args[1].get("json"))
        assert payload["msg_type"] == "interactive"
        # BUY → green card
        assert payload["card"]["header"]["template"] == "green"
        assert "NVDA" in payload["card"]["elements"][0]["content"]

    @patch("tradingagents.notify.feishu_notifier.requests.post")
    def test_feishu_sell_gets_red_card(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "msg": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        hook = NotifyHook(config={
            "notifier": "feishu",
            "notifier_config": {
                "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/test",
            },
        })

        msg = "**TSLA** on 2026-03-17\nDecision: **SELL**  |  Confidence: HIGH"
        hook._send(msg)

        payload = mock_post.call_args.kwargs.get("json", mock_post.call_args[1].get("json"))
        assert payload["card"]["header"]["template"] == "red"


class TestNotifyHookViaHookManager:
    """Full chain: HookManager.load_builtin_hooks → NotifyHook → Notifier."""

    @patch("tradingagents.notify.feishu_notifier.requests.post")
    def test_builtin_notify_with_feishu(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "msg": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        config = {
            "hooks": {
                "entries": {
                    "notify": {
                        "enabled": True,
                        "notifier": "feishu",
                        "notifier_config": {
                            "webhook_url": "https://open.feishu.cn/hook/xxx",
                        },
                    },
                }
            }
        }

        mgr = HookManager(config=config)
        mgr.load_builtin_hooks()

        assert mgr.summary["total"] == 1
        assert mgr.summary["hooks"][0]["name"] == "notify"

        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="AAPL",
            trade_date="2026-03-17",
            metadata={"decision": "HOLD", "confidence": "MEDIUM"},
        )
        mgr.dispatch(ctx)

        # Give the daemon thread time to fire
        time.sleep(0.3)

        mock_post.assert_called_once()

    def test_log_backend_does_not_crash(self):
        """notifier=log works without any external dependency."""
        hook = NotifyHook(config={"notifier": "log"})
        ctx = HookContext(
            event=HookEvent.AFTER_DECISION,
            ticker="GOOG",
            trade_date="2026-03-17",
            metadata={"decision": "BUY", "confidence": "LOW"},
        )
        # Should not raise
        result = hook.handle(ctx)
        assert result.event == HookEvent.AFTER_DECISION
