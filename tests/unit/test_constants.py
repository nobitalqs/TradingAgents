from tradingagents.constants import (
    ANALYST_TYPES,
    VALID_SIGNALS,
    analyst_node_name,
    msg_clear_node_name,
    tools_node_name,
)


def test_analyst_types_count():
    assert len(ANALYST_TYPES) == 4


def test_analyst_types_members():
    for t in ("market", "social", "news", "fundamentals"):
        assert t in ANALYST_TYPES


def test_valid_signals():
    assert set(VALID_SIGNALS) == {"BUY", "SELL", "HOLD"}


def test_analyst_node_name():
    assert analyst_node_name("market") == "Market Analyst"
    assert analyst_node_name("fundamentals") == "Fundamentals Analyst"


def test_msg_clear_node_name():
    assert msg_clear_node_name("news") == "Msg Clear News"


def test_tools_node_name():
    assert tools_node_name("social") == "tools_social"
