"""Reflection system — 5 reflectors update memories from outcomes.

Shared reflect_memories() logic used by both manual reflect_and_remember()
and the AutoReflector (T+N verification).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("tradingagents.graph.reflection")

_REFLECTION_PROMPT = """You are an expert financial analyst reviewing trading decisions. Provide detailed analysis:

1. REASONING: Was the decision correct (increased returns) or incorrect? Analyze factors:
   - Market intelligence and technical indicators
   - Price movement analysis
   - News and sentiment analysis
   - Fundamental data
   - Weight each factor's importance

2. IMPROVEMENT: For incorrect decisions, propose corrections with specific recommendations.

3. SUMMARY: Lessons learned and how to apply them to similar future situations.

4. KEY INSIGHT: Condense lessons into a concise sentence (<500 tokens) for memory storage."""


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, llm):
        self.llm = llm

    def _extract_situation(self, state: dict[str, Any]) -> str:
        """Extract market situation summary from state."""
        parts = [
            state.get("market_report", ""),
            state.get("sentiment_report", ""),
            state.get("news_report", ""),
            state.get("fundamentals_report", ""),
        ]
        return "\n\n".join(p for p in parts if p)

    def _reflect_on_component(
        self, component: str, report: str, situation: str, returns_losses: Any
    ) -> str:
        """Generate reflection for one component."""
        messages = [
            ("system", _REFLECTION_PROMPT),
            ("human", (
                f"Component: {component}\n"
                f"Returns/Losses: {returns_losses}\n\n"
                f"Analysis/Decision:\n{report}\n\n"
                f"Market Context:\n{situation}"
            )),
        ]
        try:
            return self.llm.invoke(messages).content
        except Exception as e:
            logger.error(f"Reflection failed for {component}: {e}")
            return f"Reflection failed: {e}"

    def reflect_bull(self, state: dict, returns_losses: Any, memory) -> None:
        situation = self._extract_situation(state)
        debate = state.get("investment_debate_state", {})
        history = debate.get("bull_history", [])
        report = "\n".join(history) if isinstance(history, list) else str(history)
        result = self._reflect_on_component("BULL", report, situation, returns_losses)
        memory.add_situations([(situation, result)])

    def reflect_bear(self, state: dict, returns_losses: Any, memory) -> None:
        situation = self._extract_situation(state)
        debate = state.get("investment_debate_state", {})
        history = debate.get("bear_history", [])
        report = "\n".join(history) if isinstance(history, list) else str(history)
        result = self._reflect_on_component("BEAR", report, situation, returns_losses)
        memory.add_situations([(situation, result)])

    def reflect_trader(self, state: dict, returns_losses: Any, memory) -> None:
        situation = self._extract_situation(state)
        report = state.get("trader_investment_plan", "")
        result = self._reflect_on_component("TRADER", report, situation, returns_losses)
        memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, state: dict, returns_losses: Any, memory) -> None:
        situation = self._extract_situation(state)
        debate = state.get("investment_debate_state", {})
        report = debate.get("judge_decision", "")
        result = self._reflect_on_component("INVEST_JUDGE", report, situation, returns_losses)
        memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, state: dict, returns_losses: Any, memory) -> None:
        situation = self._extract_situation(state)
        debate = state.get("risk_debate_state", {})
        report = debate.get("judge_decision", "")
        result = self._reflect_on_component("RISK_JUDGE", report, situation, returns_losses)
        memory.add_situations([(situation, result)])


def reflect_memories(
    reflector: Reflector,
    state: dict,
    returns_losses: Any,
    memories: dict[str, Any],
) -> None:
    """Shared reflection logic — calls all 5 reflectors.

    Args:
        reflector: Reflector instance with LLM
        state: Full agent state from last propagation
        returns_losses: Return/loss data for this trade
        memories: Dict with keys: bull_memory, bear_memory, trader_memory,
                  invest_judge_memory, risk_manager_memory
    """
    reflector.reflect_bull(state, returns_losses, memories["bull_memory"])
    reflector.reflect_bear(state, returns_losses, memories["bear_memory"])
    reflector.reflect_trader(state, returns_losses, memories["trader_memory"])
    reflector.reflect_invest_judge(state, returns_losses, memories["invest_judge_memory"])
    reflector.reflect_risk_manager(state, returns_losses, memories["risk_manager_memory"])
    logger.info("Reflection complete for all 5 components")
