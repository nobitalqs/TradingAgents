"""Research Manager — investment debate judge with structured scoring."""

import logging

logger = logging.getLogger("tradingagents.agents.managers.research")


def create_research_manager(llm, memory):
    """Create the Research Manager (Investment Judge) node.

    Evaluates bull/bear debate with 3-dimensional scoring,
    weighted by data credibility.
    """

    def research_manager_node(state: dict) -> dict:
        debate = state["investment_debate_state"]
        history = debate.get("history", [])

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
                "\n\nDATA CREDIBILITY CONTEXT — weight your decision toward verified data:\n"
                + "\n".join(f"- {w}" for w in cred["warnings"])
            )

        # Consensus context
        consensus_note = ""
        consensus = state.get("analyst_consensus", {})
        if consensus:
            consensus_note = (
                f"\n\nAnalyst consensus: "
                f"BUY={consensus.get('buy_count', 0)} "
                f"SELL={consensus.get('sell_count', 0)} "
                f"HOLD={consensus.get('hold_count', 0)} "
                f"(confidence: {consensus.get('confidence', 'N/A')})"
            )

        prompt = f"""As the Research Manager and debate judge, evaluate the bull vs bear debate and make a definitive decision.

SCORING FRAMEWORK (rate each 1-10):
1. Evidence Quality: How well-supported are the arguments? Penalize unverified data.
2. Risk-Reward Ratio: Does the potential upside justify the risks?
3. Timing Appropriateness: Is now the right time given market conditions?

DECISION RULES:
- Score difference >= 6: Strong conviction toward winner
- Score difference 3-5: Moderate conviction, consider partial position
- Score difference <= 2: Split decision, lean toward HOLD

Provide:
1. Your recommendation: BUY, SELL, or HOLD (do not default to HOLD without strong justification)
2. Rationale tied to the 3 scoring dimensions
3. A detailed investment plan with strategic actions

Past reflections on mistakes:
{past_memory_str or "(No past reflections)"}
{credibility_note}
{consensus_note}

Debate History:
{_format_history(history)}"""

        response = llm.invoke(prompt)

        return {
            "investment_debate_state": {
                "history": history,
                "bull_history": debate.get("bull_history", []),
                "bear_history": debate.get("bear_history", []),
                "current_response": response.content,
                "judge_decision": response.content,
                "count": debate["count"],
            },
            "investment_plan": response.content,
        }

    return research_manager_node


def _format_history(history: list[str]) -> str:
    if not history:
        return "(No debate history)"
    return "\n\n---\n\n".join(history)
