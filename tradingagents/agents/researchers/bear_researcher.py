"""Bear Researcher — advocates against investing, with fact-checking responsibility."""

import logging

logger = logging.getLogger("tradingagents.agents.researchers.bear")


def create_bear_researcher(llm, memory):
    """Create the Bear Researcher node.

    The bear researcher builds risk cases while also
    fact-checking the bull's data sources for reliability.
    """

    def bear_node(state: dict) -> dict:
        debate = state["investment_debate_state"]
        history = debate.get("history", [])
        bear_history = debate.get("bear_history", [])

        current_response = debate.get("current_response", "")
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
        )

        # Credibility context
        credibility_note = ""
        cred = state.get("data_credibility", {})
        if cred.get("warnings"):
            credibility_note = (
                "\n\nDATA CREDIBILITY WARNINGS:\n"
                + "\n".join(f"- {w}" for w in cred["warnings"])
            )

        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Present well-reasoned arguments emphasizing risks, challenges, and negative indicators.

Key responsibilities:
1. RISKS: Market saturation, financial instability, macroeconomic threats
2. COMPETITIVE WEAKNESSES: Weaker positioning, declining innovation, competitor threats
3. NEGATIVE INDICATORS: Financial data, market trends, adverse news
4. COUNTER BULL: Expose weaknesses or over-optimistic assumptions in bull arguments
5. FACT-CHECK: Challenge any data or news cited by the Bull that comes from unverified sources. If Bull relies on a single news source for a key claim, flag as "SINGLE-SOURCE RISK". Cross-reference key claims against verified data (price action, SEC filings)
6. LEARN FROM PAST: Apply lessons from similar past situations

Resources:
Market report: {market_report}
Sentiment report: {sentiment_report}
News report: {news_report}
Fundamentals report: {fundamentals_report}
Debate history: {_format_history(history)}
Last bull argument: {current_response}
Past lessons: {past_memory_str}
{credibility_note}

Engage conversationally — debate the bull's points directly, don't just list facts."""

        response = llm.invoke(prompt)
        argument = f"Bear Analyst: {response.content}"

        return {
            "investment_debate_state": {
                "history": history + [argument],
                "bear_history": bear_history + [argument],
                "bull_history": debate.get("bull_history", []),
                "current_response": argument,
                "judge_decision": debate.get("judge_decision", ""),
                "count": debate["count"] + 1,
            }
        }

    return bear_node


def _format_history(history: list[str]) -> str:
    """Format debate history for prompt injection."""
    if not history:
        return "(No debate history yet)"
    return "\n\n".join(history[-4:])
