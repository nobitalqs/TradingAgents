"""Bull Researcher — advocates for investing, with fact-checking responsibility."""

import logging

logger = logging.getLogger("tradingagents.agents.researchers.bull")


def create_bull_researcher(llm, memory):
    """Create the Bull Researcher node.

    The bull researcher builds investment cases while also
    fact-checking the bear's data sources for reliability.
    """

    def bull_node(state: dict) -> dict:
        debate = state["investment_debate_state"]
        history = debate.get("history", [])
        bull_history = debate.get("bull_history", [])

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

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators.

Key responsibilities:
1. GROWTH CASE: Highlight market opportunities, revenue projections, scalability
2. COMPETITIVE ADVANTAGES: Unique products, strong branding, market positioning
3. POSITIVE INDICATORS: Financial health, industry trends, positive news
4. COUNTER BEAR: Address bear arguments with specific data and reasoning
5. FACT-CHECK: Challenge any data or news cited by the Bear that comes from unverified sources. If Bear relies on a single news source for a key claim, flag as "SINGLE-SOURCE RISK"
6. LEARN FROM PAST: Apply lessons from similar past situations

Resources:
Market report: {market_report}
Sentiment report: {sentiment_report}
News report: {news_report}
Fundamentals report: {fundamentals_report}
Debate history: {_format_history(history)}
Last bear argument: {current_response}
Past lessons: {past_memory_str}
{credibility_note}

Engage conversationally — debate the bear's points directly, don't just list facts."""

        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"

        return {
            "investment_debate_state": {
                "history": history + [argument],
                "bull_history": bull_history + [argument],
                "bear_history": debate.get("bear_history", []),
                "current_response": argument,
                "judge_decision": debate.get("judge_decision", ""),
                "count": debate["count"] + 1,
            }
        }

    return bull_node


def _format_history(history: list[str]) -> str:
    """Format debate history for prompt injection."""
    if not history:
        return "(No debate history yet)"
    return "\n\n".join(history[-4:])  # last 4 turns to stay within context
