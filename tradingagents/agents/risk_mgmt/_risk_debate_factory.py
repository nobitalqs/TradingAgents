"""Shared factory for risk debate participants — eliminates copy-paste."""

import logging

logger = logging.getLogger("tradingagents.agents.risk_mgmt")


def create_risk_debator(
    llm,
    stance: str,
    system_message: str,
    history_key: str,
) -> callable:
    """Create a risk debater node.

    Args:
        llm: LangChain LLM instance.
        stance: "Aggressive", "Conservative", or "Neutral".
        system_message: Stance-specific system prompt.
        history_key: Key in risk_debate_state for this debater's history.

    Returns:
        Node function: (state) -> dict
    """

    def debator_node(state: dict) -> dict:
        risk_debate = state["risk_debate_state"]
        history = risk_debate.get("history", [])
        own_history = risk_debate.get(history_key, [])

        trader_decision = state.get("trader_investment_plan", "")

        prompt = f"""{system_message}

Trader's decision to evaluate:
{trader_decision}

Market report: {state.get('market_report', '')}
Sentiment report: {state.get('sentiment_report', '')}
News report: {state.get('news_report', '')}
Fundamentals report: {state.get('fundamentals_report', '')}

Debate history (last 4 turns):
{_format_history(history)}

Present your arguments conversationally. No special formatting."""

        response = llm.invoke(prompt)
        argument = f"{stance} Analyst: {response.content}"

        new_state = {
            "history": history + [argument],
            history_key: own_history + [argument],
            "latest_speaker": stance,
            "judge_decision": risk_debate.get("judge_decision", ""),
            "count": risk_debate["count"] + 1,
        }

        # Preserve other debaters' histories
        for key in ("aggressive_history", "conservative_history", "neutral_history"):
            if key != history_key:
                new_state[key] = risk_debate.get(key, [])

        return {"risk_debate_state": new_state}

    return debator_node


def _format_history(history: list[str]) -> str:
    if not history:
        return "(No debate history yet)"
    return "\n\n".join(history[-4:])
