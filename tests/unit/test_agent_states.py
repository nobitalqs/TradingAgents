from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
    create_empty_invest_debate_state,
    create_empty_risk_debate_state,
)


def test_create_empty_invest_debate():
    state = create_empty_invest_debate_state()
    assert state["count"] == 0
    assert state["judge_decision"] == ""
    assert state["bull_history"] == []
    assert state["bear_history"] == []
    assert state["history"] == []
    assert state["current_response"] == ""


def test_create_empty_risk_debate():
    state = create_empty_risk_debate_state()
    assert state["count"] == 0
    assert state["aggressive_history"] == []
    assert state["conservative_history"] == []
    assert state["neutral_history"] == []
    assert state["latest_speaker"] == ""


def test_agent_state_has_required_fields():
    annotations = AgentState.__annotations__
    required = [
        "messages", "company_of_interest", "trade_date",
        "market_report", "sentiment_report", "news_report", "fundamentals_report",
        "investment_debate_state", "investment_plan",
        "trader_investment_plan", "risk_debate_state", "final_trade_decision",
        "analyst_consensus", "market_regime", "data_credibility",
    ]
    for field in required:
        assert field in annotations, f"Missing field: {field}"


def test_invest_debate_state_is_typed_dict():
    assert issubclass(InvestDebateState, dict)


def test_risk_debate_state_is_typed_dict():
    assert issubclass(RiskDebateState, dict)
