"""Agent state definitions — single source of truth for all graph state."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class InvestDebateState(TypedDict):
    """Bull/Bear investment debate state."""

    history: list[str]
    bull_history: list[str]
    bear_history: list[str]
    current_response: str
    judge_decision: str
    count: int


class RiskDebateState(TypedDict):
    """Aggressive/Conservative/Neutral risk debate state."""

    history: list[str]
    aggressive_history: list[str]
    conservative_history: list[str]
    neutral_history: list[str]
    latest_speaker: str
    judge_decision: str
    count: int


class AgentState(MessagesState):
    """Main graph state — all agents read/write through this."""

    company_of_interest: Annotated[str, "Ticker symbol"]
    trade_date: Annotated[str, "Analysis date"]
    sender: Annotated[str, "Last agent that sent a message"]

    # Analyst reports
    market_report: Annotated[str, "Technical analysis report"]
    sentiment_report: Annotated[str, "Social sentiment report"]
    news_report: Annotated[str, "News and macro report"]
    fundamentals_report: Annotated[str, "Financial fundamentals report"]

    # Consensus & regime (enhanced)
    analyst_consensus: Annotated[dict, "Consensus voting from all analysts"]
    market_regime: Annotated[dict, "Market regime classification"]
    data_credibility: Annotated[dict, "Data credibility summary"]

    # Debate states
    investment_debate_state: Annotated[
        InvestDebateState, "Bull/Bear debate state"
    ]
    investment_plan: Annotated[str, "Investment plan from research manager"]
    trader_investment_plan: Annotated[str, "Execution plan from trader"]

    # Risk management
    risk_debate_state: Annotated[
        RiskDebateState, "Risk debate state"
    ]
    final_trade_decision: Annotated[str, "Final decision from risk judge"]


def create_empty_invest_debate_state() -> InvestDebateState:
    """Create a fresh invest debate state."""
    return InvestDebateState(
        history=[],
        bull_history=[],
        bear_history=[],
        current_response="",
        judge_decision="",
        count=0,
    )


def create_empty_risk_debate_state() -> RiskDebateState:
    """Create a fresh risk debate state."""
    return RiskDebateState(
        history=[],
        aggressive_history=[],
        conservative_history=[],
        neutral_history=[],
        latest_speaker="",
        judge_decision="",
        count=0,
    )
