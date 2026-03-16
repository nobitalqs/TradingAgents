"""Generic webhook notifier."""

import logging

import requests

from tradingagents.notify.base import BaseNotifier

logger = logging.getLogger(__name__)


class WebhookNotifier(BaseNotifier):
    """Send notifications to a generic HTTP webhook endpoint."""

    def send(self, message: str) -> bool:
        """POST ``{"text": message, "source": "TradingAgents"}`` to the URL.

        Returns True when the HTTP response status is 200.
        """
        url = self.config.get("url", "")
        if not url:
            logger.error("Webhook url is empty; cannot send notification.")
            return False

        try:
            resp = requests.post(
                url,
                json={"text": message, "source": "TradingAgents"},
                timeout=10,
            )
            success = resp.status_code == 200
            if not success:
                logger.warning(
                    "Webhook returned status %s: %s",
                    resp.status_code,
                    resp.text,
                )
            return success
        except Exception:
            logger.exception("Failed to send webhook notification.")
            return False
