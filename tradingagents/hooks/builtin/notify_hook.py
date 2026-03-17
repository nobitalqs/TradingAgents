"""NotifyHook: non-blocking notifications for decisions and alerts.

Dispatches to real notifier backends (Feishu, Slack, Webhook) from
the ``tradingagents.notify`` package.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent
from tradingagents.notify.base import BaseNotifier

logger = logging.getLogger(__name__)

# Registry: notifier name → (module_path, class_name)
_NOTIFIER_REGISTRY: dict[str, tuple[str, str]] = {
    "feishu": ("tradingagents.notify.feishu_notifier", "FeishuNotifier"),
    "slack": ("tradingagents.notify.slack_notifier", "SlackNotifier"),
    "webhook": ("tradingagents.notify.webhook_notifier", "WebhookNotifier"),
}


def _create_notifier(name: str, config: dict[str, Any]) -> BaseNotifier | None:
    """Lazily import and instantiate a notifier backend by name."""
    entry = _NOTIFIER_REGISTRY.get(name)
    if entry is None:
        logger.warning("Unknown notifier backend: %s", name)
        return None

    module_path, class_name = entry
    try:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(config=config)
    except Exception:
        logger.exception("Failed to create notifier %s", name)
        return None


class NotifyHook(BaseHook):
    """Send notifications asynchronously via a background thread.

    Config keys::

        {
            "enabled": true,
            "notifier": "feishu",          # feishu | slack | webhook | log
            "notifier_config": {
                "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
                "secret": "optional-signing-secret"
            }
        }
    """

    name = "notify"
    subscriptions = [HookEvent.AFTER_DECISION, HookEvent.HEARTBEAT_ALERT]

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        self._notifier: BaseNotifier | None = None

    def _get_notifier(self) -> BaseNotifier | None:
        """Lazy-init the notifier on first use."""
        if self._notifier is not None:
            return self._notifier

        notifier_name = self.config.get("notifier", "log")
        if notifier_name == "log":
            return None

        notifier_config = self.config.get("notifier_config", {})
        self._notifier = _create_notifier(notifier_name, notifier_config)
        return self._notifier

    def handle(self, context: HookContext) -> HookContext:
        message = self._format_message(context)
        if not message:
            return context

        thread = threading.Thread(
            target=self._send,
            args=(message,),
            daemon=True,
        )
        thread.start()
        return context

    # ── message formatting ───────────────────────────────────────

    def _format_message(self, context: HookContext) -> str:
        if context.event == HookEvent.AFTER_DECISION:
            decision = context.metadata.get("decision", "unknown")
            confidence = context.metadata.get("confidence", "N/A")
            full_signal = context.metadata.get("full_signal", "")

            lines = [
                f"**{context.ticker}** on {context.trade_date}",
                f"Decision: **{decision}**  |  Confidence: {confidence}",
            ]
            if full_signal:
                # Truncate long decisions for readability
                preview = full_signal[:500] + ("..." if len(full_signal) > 500 else "")
                lines.append(f"\n{preview}")
            return "\n".join(lines)

        if context.event == HookEvent.HEARTBEAT_ALERT:
            alert_type = context.metadata.get("type", "unknown")
            ticker = context.metadata.get("ticker", "")
            change = context.metadata.get("change_pct", context.metadata.get("ratio", ""))
            return f"**[ALERT]** {ticker} — {alert_type} ({change})"

        return ""

    # ── send (runs in background thread) ─────────────────────────

    def _send(self, message: str) -> None:
        notifier_name = self.config.get("notifier", "log")

        if notifier_name == "log":
            logger.info("Notification: %s", message)
            return

        notifier = self._get_notifier()
        if notifier is None:
            logger.warning("No notifier available for backend: %s", notifier_name)
            return

        try:
            success = notifier.send(message)
            if success:
                logger.info("Notification sent via %s", notifier_name)
            else:
                logger.warning("Notification via %s returned failure", notifier_name)
        except Exception:
            logger.exception("Failed to send notification via %s", notifier_name)
