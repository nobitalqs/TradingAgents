"""Shared test fixtures."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """Mock LLM that returns configurable responses."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "BUY"
    response.tool_calls = []
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def sample_state():
    """Minimal AgentState for testing."""
    from tradingagents.agents.utils.agent_states import (
        create_empty_invest_debate_state,
        create_empty_risk_debate_state,
    )
    return {
        "messages": [],
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "sender": "",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "analyst_consensus": {},
        "market_regime": {},
        "data_credibility": {},
        "investment_debate_state": create_empty_invest_debate_state(),
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": create_empty_risk_debate_state(),
        "final_trade_decision": "",
    }
