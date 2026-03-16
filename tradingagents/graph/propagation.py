"""State initialization and graph invocation configuration."""

from __future__ import annotations

from typing import Any

from tradingagents.agents.utils.agent_states import (
    create_empty_invest_debate_state,
    create_empty_risk_debate_state,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit: int = 100):
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph."""
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "sender": "",
            # Analyst reports
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            # Consensus & regime (new fields)
            "analyst_consensus": {},
            "market_regime": {},
            "data_credibility": {},
            # Debate states
            "investment_debate_state": create_empty_invest_debate_state(),
            "investment_plan": "",
            "trader_investment_plan": "",
            "risk_debate_state": create_empty_risk_debate_state(),
            "final_trade_decision": "",
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation."""
        config: dict[str, Any] = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
