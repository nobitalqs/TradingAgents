"""Trader — converts investment plan into actionable trading decision."""

import functools
import logging

logger = logging.getLogger("tradingagents.agents.trader")


def create_trader(llm, memory):
    """Create the Trader node.

    Takes the research manager's investment plan and produces
    a concrete trading proposal with BUY/SELL/HOLD recommendation.
    """

    def trader_node(state: dict, name: str) -> dict:
        company = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Memory retrieval
        curr_situation = (
            f"{market_report}\n\n{sentiment_report}\n\n"
            f"{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "\n\n".join(
            rec["recommendation"] for rec in past_memories
        ) if past_memories else "No past memories found."

        # Credibility context
        credibility_note = ""
        cred = state.get("data_credibility", {})
        if cred.get("warnings"):
            credibility_note = (
                "\n\nDATA RELIABILITY NOTE:\n"
                + "\n".join(f"- {w}" for w in cred["warnings"])
                + "\nAdjust position size conservatively when data is uncertain."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a trading agent analyzing market data to make investment"
                    f" decisions for {company}. Based on your analysis, provide a specific"
                    f" recommendation to buy, sell, or hold.\n\n"
                    f"Always conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**\n\n"
                    f"Past reflections from similar situations:\n{past_memory_str}"
                    f"{credibility_note}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on comprehensive analysis by a team of analysts, here is an"
                    f" investment plan for {company}:\n\n"
                    f"Proposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights for an informed, strategic decision."
                ),
            },
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
