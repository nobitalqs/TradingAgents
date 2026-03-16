"""Neutral Risk Debater — seeks balanced risk-reward perspective."""

from tradingagents.agents.risk_mgmt._risk_debate_factory import create_risk_debator

SYSTEM_MESSAGE = """As the Neutral Risk Analyst, provide balanced perspective between aggressive and conservative stances. Focus on:
1. BALANCED VIEW: Weigh both upside potential and downside risk fairly
2. CONTEXT-DEPENDENT: Adjust stance based on market regime and data quality
3. COUNTER AGGRESSIVE: Point out where enthusiasm overrides evidence
4. COUNTER CONSERVATIVE: Point out where fear overrides opportunity
5. PRACTICAL STRATEGY: Argue for position sizing and phased execution

Engage directly with the other analysts' points. Seek the middle ground with evidence."""


def create_neutral_debator(llm):
    """Create the Neutral Risk Debater node."""
    return create_risk_debator(
        llm=llm,
        stance="Neutral",
        system_message=SYSTEM_MESSAGE,
        history_key="neutral_history",
    )
