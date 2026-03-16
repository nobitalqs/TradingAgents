import copy

from tradingagents.default_config import DEFAULT_CONFIG


def test_config_has_required_keys():
    required = [
        "llm_provider", "deep_think_llm", "quick_think_llm",
        "max_debate_rounds", "max_risk_discuss_rounds",
        "data_vendors", "hooks", "scheduler", "heartbeat", "notify",
        "verification", "message_gateway", "learning",
    ]
    for key in required:
        assert key in DEFAULT_CONFIG, f"Missing key: {key}"


def test_config_deep_copy_isolation():
    c1 = copy.deepcopy(DEFAULT_CONFIG)
    c2 = copy.deepcopy(DEFAULT_CONFIG)
    c1["hooks"]["enabled"] = False
    assert DEFAULT_CONFIG["hooks"]["enabled"] is True
    assert c2["hooks"]["enabled"] is True


def test_hooks_default_state():
    assert DEFAULT_CONFIG["hooks"]["enabled"] is True
    assert DEFAULT_CONFIG["scheduler"]["enabled"] is False
    assert DEFAULT_CONFIG["heartbeat"]["enabled"] is False


def test_data_vendors_defaults():
    vendors = DEFAULT_CONFIG["data_vendors"]
    assert vendors["core_stock_apis"] == "yfinance"
    assert vendors["news_data"] == "yfinance"


def test_verification_defaults():
    v = DEFAULT_CONFIG["verification"]
    assert v["enabled"] is True
    assert v["news_credibility_scoring"] is True
