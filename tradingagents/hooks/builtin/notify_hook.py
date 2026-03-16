"""NotifyHook: non-blocking notifications for decisions and alerts."""

from __future__ import annotations

import logging
import threading

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent

logger = logging.getLogger(__name__)


class NotifyHook(BaseHook):
    """Send notifications asynchronously via a background thread.

    Config keys:
        notifier (str): notifier backend name (e.g. "slack", "email").
        notifier_config (dict): backend-specific settings.
    """

    name = "notify"
    subscriptions = [HookEvent.AFTER_DECISION, HookEvent.HEARTBEAT_ALERT]

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
            return (
                f"[TRADE DECISION] {context.ticker} on {context.trade_date}: "
                f"{decision} (confidence: {confidence})"
            )

        if context.event == HookEvent.HEARTBEAT_ALERT:
            alert = context.metadata.get("alert_message", "Unknown alert")
            return f"[ALERT] {alert}"

        return ""

    # ── send (runs in background thread) ─────────────────────────

    def _send(self, message: str) -> None:
        notifier_name = self.config.get("notifier", "log")

        if notifier_name == "log":
            logger.info("Notification: %s", message)
            return

        try:
            # Lazy import of notifier backends
            if notifier_name == "slack":
                from tradingagents.hooks.builtin._notifiers import slack as backend
            elif notifier_name == "email":
                from tradingagents.hooks.builtin._notifiers import email as backend
            else:
                logger.warning("Unknown notifier backend: %s", notifier_name)
                return

            backend.send(message, self.config.get("notifier_config", {}))
        except ImportError:
            logger.error(
                "Notifier backend %r not available",
                notifier_name,
                exc_info=True,
            )
        except Exception:
            logger.error(
                "Failed to send notification via %s",
                notifier_name,
                exc_info=True,
            )
