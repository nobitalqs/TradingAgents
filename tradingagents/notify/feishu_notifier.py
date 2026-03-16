"""Feishu (Lark) webhook notifier with interactive card support."""

import base64
import hashlib
import hmac
import logging
import time

import requests

from tradingagents.notify.base import BaseNotifier

logger = logging.getLogger(__name__)

# Action keyword -> (color template, header title)
_ACTION_STYLES: dict[str, tuple[str, str]] = {
    "BUY": ("green", "Buy Signal"),
    "SELL": ("red", "Sell Signal"),
    "HOLD": ("blue", "Hold Signal"),
}

_ALERT_KEYWORDS: list[str] = [
    "alert",
    "warning",
    "warn",
    "error",
    "critical",
    "urgent",
]


def _detect_action(message: str) -> tuple[str, str]:
    """Return (template_color, header_title) based on message content."""
    upper = message.upper()
    for keyword, (color, title) in _ACTION_STYLES.items():
        if keyword in upper:
            return color, title

    lower = message.lower()
    for kw in _ALERT_KEYWORDS:
        if kw in lower:
            return "orange", "Alert"

    return "blue", "Notification"


class FeishuNotifier(BaseNotifier):
    """Send notifications to Feishu via incoming webhook."""

    def send(self, message: str) -> bool:
        """POST a card message to the configured Feishu webhook.

        Returns True when the Feishu API responds with code == 0.
        """
        webhook_url = self.config.get("webhook_url", "")
        if not webhook_url:
            logger.error("Feishu webhook_url is empty; cannot send notification.")
            return False

        payload = self._build_card_message(message)

        secret = self.config.get("secret", "")
        if secret:
            timestamp = str(int(time.time()))
            sign = self._gen_sign(timestamp, secret)
            payload["timestamp"] = timestamp
            payload["sign"] = sign

        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            success = data.get("code", -1) == 0
            if not success:
                logger.warning("Feishu API returned non-zero code: %s", data)
            return success
        except Exception:
            logger.exception("Failed to send Feishu notification.")
            return False

    def _build_card_message(self, message: str) -> dict:
        """Build a Feishu interactive card payload.

        Card colours:
        - green  for BUY
        - red    for SELL
        - blue   for HOLD / default
        - orange for alert / warning keywords
        """
        template, title = _detect_action(message)

        card: dict = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "template": template,
                    "title": {
                        "tag": "plain_text",
                        "content": title,
                    },
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": message,
                    },
                    {
                        "tag": "hr",
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": "Sent by TradingAgents",
                            }
                        ],
                    },
                ],
            },
        }
        return card

    @staticmethod
    def _gen_sign(timestamp: str, secret: str) -> str:
        """Generate HMAC-SHA256 base64 signature for Feishu webhook.

        The signing string is ``"{timestamp}\\n{secret}"`` as required by
        the Feishu bot security spec.
        """
        string_to_sign = f"{timestamp}\n{secret}"
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        return base64.b64encode(hmac_code).decode("utf-8")
