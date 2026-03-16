"""Slack incoming webhook notifier."""

import logging

import requests

from tradingagents.notify.base import BaseNotifier

logger = logging.getLogger(__name__)


class SlackNotifier(BaseNotifier):
    """Send plain-text notifications to a Slack incoming webhook."""

    def send(self, message: str) -> bool:
        """POST ``{"text": message}`` to the Slack webhook URL.

        Returns True when the HTTP response status is 200.
        """
        webhook_url = self.config.get("webhook_url", "")
        if not webhook_url:
            logger.error("Slack webhook_url is empty; cannot send notification.")
            return False

        try:
            resp = requests.post(
                webhook_url,
                json={"text": message},
                timeout=10,
            )
            success = resp.status_code == 200
            if not success:
                logger.warning(
                    "Slack webhook returned status %s: %s",
                    resp.status_code,
                    resp.text,
                )
            return success
        except Exception:
            logger.exception("Failed to send Slack notification.")
            return False
