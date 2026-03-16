"""Conditional logic for graph flow control — factory-generated routers."""

from __future__ import annotations

import logging

from tradingagents.constants import msg_clear_node_name, tools_node_name

logger = logging.getLogger("tradingagents.graph.conditional")


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds: int = 1, max_risk_discuss_rounds: int = 1):
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def make_analyst_router(self, analyst_type: str) -> callable:
        """Factory-generated router for analyst tool loops.

        Returns a function that checks if the last message has tool_calls.
        If yes → route to tools node; if no → route to message clear node.
        """
        tools_name = tools_node_name(analyst_type)
        clear_name = msg_clear_node_name(analyst_type)

        def router(state: dict) -> str:
            messages = state.get("messages", [])
            if not messages:
                return clear_name
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return tools_name
            return clear_name

        return router

    def should_continue_debate(self, state: dict) -> str:
        """Determine if investment debate should continue."""
        debate = state["investment_debate_state"]
        if debate["count"] >= 2 * self.max_debate_rounds:
            return "Research Manager"
        current = debate.get("current_response", "")
        if isinstance(current, str) and current.startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: dict) -> str:
        """Determine if risk analysis debate should continue."""
        risk = state["risk_debate_state"]
        if risk["count"] >= 3 * self.max_risk_discuss_rounds:
            return "Risk Judge"
        speaker = risk.get("latest_speaker", "")
        if speaker.startswith("Aggressive"):
            return "Conservative Analyst"
        if speaker.startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
