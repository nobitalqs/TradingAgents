"""Tests for the notification layer."""

from unittest.mock import MagicMock, patch

from tradingagents.notify.feishu_notifier import FeishuNotifier
from tradingagents.notify.slack_notifier import SlackNotifier
from tradingagents.notify.webhook_notifier import WebhookNotifier


# ── Feishu card colour tests ─────────────────────────────────────────


class TestFeishuCardColours:
    """Verify card header template colour matches the trading action."""

    def _card_template(self, message: str) -> str:
        notifier = FeishuNotifier({"webhook_url": "https://example.com/hook"})
        card = notifier._build_card_message(message)
        return card["card"]["header"]["template"]

    def test_feishu_card_green_on_buy(self):
        assert self._card_template("Recommendation: BUY AAPL") == "green"

    def test_feishu_card_red_on_sell(self):
        assert self._card_template("Signal: SELL TSLA immediately") == "red"

    def test_feishu_card_blue_on_hold(self):
        assert self._card_template("Maintain HOLD position on MSFT") == "blue"

    def test_feishu_card_orange_on_alert(self):
        assert self._card_template("warning: unusual volume detected") == "orange"


# ── Feishu signature test ────────────────────────────────────────────


class TestFeishuSignature:
    def test_feishu_sign_generation(self):
        sig = FeishuNotifier._gen_sign("1700000000", "test-secret")
        assert isinstance(sig, str)
        assert len(sig) > 0


# ── Feishu send tests (mocked) ──────────────────────────────────────


class TestFeishuSend:
    @patch("tradingagents.notify.feishu_notifier.requests.post")
    def test_send_success(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        notifier = FeishuNotifier({"webhook_url": "https://feishu.example/hook"})
        assert notifier.send("BUY AAPL") is True
        mock_post.assert_called_once()

    @patch("tradingagents.notify.feishu_notifier.requests.post")
    def test_send_with_secret(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        notifier = FeishuNotifier(
            {"webhook_url": "https://feishu.example/hook", "secret": "s3cret"}
        )
        assert notifier.send("HOLD GOOG") is True
        payload = mock_post.call_args.kwargs.get(
            "json", mock_post.call_args[1].get("json")
        )
        assert "timestamp" in payload
        assert "sign" in payload

    @patch("tradingagents.notify.feishu_notifier.requests.post")
    def test_send_failure_code(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 9999, "msg": "invalid"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        notifier = FeishuNotifier({"webhook_url": "https://feishu.example/hook"})
        assert notifier.send("BUY AAPL") is False

    def test_send_returns_false_no_url(self):
        notifier = FeishuNotifier({"webhook_url": ""})
        assert notifier.send("test") is False


# ── Slack tests ──────────────────────────────────────────────────────


class TestSlackNotifier:
    def test_slack_returns_false_no_url(self):
        notifier = SlackNotifier({"webhook_url": ""})
        assert notifier.send("x") is False

    @patch("tradingagents.notify.slack_notifier.requests.post")
    def test_send_success(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notifier = SlackNotifier({"webhook_url": "https://hooks.slack.com/test"})
        assert notifier.send("Hello Slack") is True
        mock_post.assert_called_once_with(
            "https://hooks.slack.com/test",
            json={"text": "Hello Slack"},
            timeout=10,
        )

    @patch("tradingagents.notify.slack_notifier.requests.post")
    def test_send_failure(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_post.return_value = mock_resp

        notifier = SlackNotifier({"webhook_url": "https://hooks.slack.com/test"})
        assert notifier.send("Hello Slack") is False


# ── Webhook tests ────────────────────────────────────────────────────


class TestWebhookNotifier:
    def test_webhook_returns_false_no_url(self):
        notifier = WebhookNotifier({"url": ""})
        assert notifier.send("x") is False

    @patch("tradingagents.notify.webhook_notifier.requests.post")
    def test_send_success(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notifier = WebhookNotifier({"url": "https://example.com/webhook"})
        assert notifier.send("payload") is True
        mock_post.assert_called_once_with(
            "https://example.com/webhook",
            json={"text": "payload", "source": "TradingAgents"},
            timeout=10,
        )

    @patch("tradingagents.notify.webhook_notifier.requests.post")
    def test_send_failure(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        notifier = WebhookNotifier({"url": "https://example.com/webhook"})
        assert notifier.send("payload") is False
