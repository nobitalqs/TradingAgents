"""Graph setup — assembles the LangGraph workflow with all nodes and edges.

Architecture:
  START → [Analysts sequential] → Extract Signals → [Invest Debate] →
  Research Manager → Trader → [Risk Debate] → Risk Judge → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from langgraph.prebuilt import ToolNode

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.managers.risk_manager import create_risk_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import create_msg_delete
from tradingagents.constants import analyst_node_name, msg_clear_node_name, tools_node_name
from tradingagents.graph.analyst_signals import create_extract_signals_node
from tradingagents.graph.conditional_logic import ConditionalLogic

logger = logging.getLogger("tradingagents.graph.setup")

# Retry policy for LLM-backed nodes — survives transient 429/timeout/5xx.
NODE_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=10.0,
    backoff_factor=2.0,
    max_interval=120.0,
    jitter=True,
)

# Registry: analyst_type → creator function
ANALYST_REGISTRY: dict[str, callable] = {
    "market": create_market_analyst,
    "social": create_social_media_analyst,
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
}


class GraphSetup:
    """Assembles and compiles the full LangGraph workflow."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        self.quick_llm = quick_thinking_llm
        self.deep_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.cond = conditional_logic

    def setup_graph(
        self,
        selected_analysts: list[str] | None = None,
    ) -> Any:
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts: List of analyst types to include.
                Defaults to all 4: market, social, news, fundamentals.

        Returns:
            Compiled LangGraph workflow.
        """
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]

        if not selected_analysts:
            raise ValueError("No analysts selected for graph setup")

        workflow = StateGraph(AgentState)

        # ── 1. Analyst nodes (sequential with tool loops) ──
        for analyst_type in selected_analysts:
            creator = ANALYST_REGISTRY.get(analyst_type)
            if not creator:
                raise ValueError(f"Unknown analyst type: {analyst_type}")

            node_name = analyst_node_name(analyst_type)
            clear_name = msg_clear_node_name(analyst_type)
            tools_name = tools_node_name(analyst_type)

            workflow.add_node(node_name, creator(self.quick_llm), retry=NODE_RETRY)
            workflow.add_node(clear_name, create_msg_delete())
            workflow.add_node(tools_name, self.tool_nodes[analyst_type])

        # ── 2. Extract Analyst Signals node ──
        workflow.add_node("Extract Signals", create_extract_signals_node())

        # ── 3. Investment debate nodes ──
        workflow.add_node(
            "Bull Researcher",
            create_bull_researcher(self.quick_llm, self.bull_memory),
            retry=NODE_RETRY,
        )
        workflow.add_node(
            "Bear Researcher",
            create_bear_researcher(self.quick_llm, self.bear_memory),
            retry=NODE_RETRY,
        )
        workflow.add_node(
            "Research Manager",
            create_research_manager(self.deep_llm, self.invest_judge_memory),
            retry=NODE_RETRY,
        )

        # ── 4. Trader node ──
        workflow.add_node(
            "Trader",
            create_trader(self.quick_llm, self.trader_memory),
            retry=NODE_RETRY,
        )

        # ── 5. Risk debate nodes ──
        workflow.add_node("Aggressive Analyst", create_aggressive_debator(self.quick_llm), retry=NODE_RETRY)
        workflow.add_node("Conservative Analyst", create_conservative_debator(self.quick_llm), retry=NODE_RETRY)
        workflow.add_node("Neutral Analyst", create_neutral_debator(self.quick_llm), retry=NODE_RETRY)
        workflow.add_node(
            "Risk Judge",
            create_risk_manager(self.deep_llm, self.risk_manager_memory),
            retry=NODE_RETRY,
        )

        # ── Edges: Analyst sequence ──
        first = selected_analysts[0]
        workflow.add_edge(START, analyst_node_name(first))

        for i, analyst_type in enumerate(selected_analysts):
            node_name = analyst_node_name(analyst_type)
            tools_name = tools_node_name(analyst_type)
            clear_name = msg_clear_node_name(analyst_type)

            # Analyst → [conditional] → tools OR clear
            router = self.cond.make_analyst_router(analyst_type)
            workflow.add_conditional_edges(
                node_name,
                router,
                [tools_name, clear_name],
            )
            # Tools → back to analyst (tool loop)
            workflow.add_edge(tools_name, node_name)

            # Clear → next analyst OR extract signals
            if i < len(selected_analysts) - 1:
                next_name = analyst_node_name(selected_analysts[i + 1])
                workflow.add_edge(clear_name, next_name)
            else:
                workflow.add_edge(clear_name, "Extract Signals")

        # ── Edges: Extract Signals → Debate ──
        workflow.add_edge("Extract Signals", "Bull Researcher")

        # ── Edges: Investment debate ──
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.cond.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.cond.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )

        # ── Edges: Manager → Trader → Risk debate ──
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")

        # ── Edges: Risk debate rotation ──
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.cond.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.cond.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.cond.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", END)

        compiled = workflow.compile()
        logger.info(
            f"Graph compiled: {len(selected_analysts)} analysts, "
            f"{len(compiled.nodes)} total nodes"
        )
        return compiled
