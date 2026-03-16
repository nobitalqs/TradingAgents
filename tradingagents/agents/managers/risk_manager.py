"""Risk Manager — final decision judge with credibility-weighted assessment."""

import logging

logger = logging.getLogger("tradingagents.agents.managers.risk")


def create_risk_manager(llm, memory):
    """Create the Risk Manager (Final Judge) node.

    Makes the final BUY/SELL/HOLD decision based on risk debate,
    weighted by data credibility and past reflections.
    """

    def risk_manager_node(state: dict) -> dict:
        risk_debate = state["risk_debate_state"]
        history = risk_debate.get("history", [])

        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        trader_plan = state.get("trader_investment_plan", "")

        # Memory retrieval
        curr_situation = (
            f"{market_report}\n\n{sentiment_report}\n\n"
            f"{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "\n\n".join(
            rec["recommendation"] for rec in past_memories
        )

        # Credibility context — reduce position size when data is unreliable
        credibility_note = ""
        cred = state.get("data_credibility", {})
        if cred.get("warnings"):
            credibility_note = (
                "\n\nDATA CREDIBILITY WARNING — reduce position size for low-credibility data:\n"
                + "\n".join(f"- {w}" for w in cred["warnings"])
                + "\n\nWhen data credibility is low, prefer HOLD or smaller positions."
            )

        prompt = f"""As the Risk Management Judge, evaluate the debate between Aggressive, Conservative, and Neutral analysts. Make a final recommendation.

SCORING FRAMEWORK (rate each 1-10):
1. Risk-Reward Ratio: Does upside justify downside risk?
2. Executability: How practical is the proposed strategy?
3. Market Fit: How well does the strategy match current market conditions?

DECISION RULES:
- Score difference >= 5: Adopt winning strategy
- Score difference < 5: Hybrid approach blending strongest elements
- Choose HOLD only if strongly justified, not as a fallback

Start with the trader's plan and refine based on debate insights:
Trader's plan: {trader_plan}

Past reflections on mistakes:
{past_memory_str or "(No past reflections)"}
{credibility_note}

Risk Debate History:
{_format_history(history)}

Provide a clear BUY, SELL, or HOLD decision with detailed reasoning."""

        response = llm.invoke(prompt)

        return {
            "risk_debate_state": {
                "history": history,
                "aggressive_history": risk_debate.get("aggressive_history", []),
                "conservative_history": risk_debate.get("conservative_history", []),
                "neutral_history": risk_debate.get("neutral_history", []),
                "latest_speaker": "Judge",
                "judge_decision": response.content,
                "count": risk_debate["count"],
            },
            "final_trade_decision": response.content,
        }

    return risk_manager_node


def _format_history(history: list[str]) -> str:
    if not history:
        return "(No debate history)"
    return "\n\n---\n\n".join(history)
