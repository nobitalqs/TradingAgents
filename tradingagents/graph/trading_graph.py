"""TradingAgentsGraph — main orchestrator with hook and verification integration."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
)
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.hooks.base import HookContext, HookEvent
from tradingagents.llm_clients import create_llm_client

from tradingagents.learning.persistence import MemoryStore

from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector, reflect_memories
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger("tradingagents.graph")


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework.

    Enhancements over upstream:
    - Hook system integration (BEFORE_PROPAGATE / AFTER_DECISION)
    - Data credibility injection via DataVerifier
    - 4-level signal extraction
    - Structured reflection with shared reflect_memories()
    - No wildcard imports, explicit dependencies
    """

    def __init__(
        self,
        selected_analysts: list[str] | None = None,
        debug: bool = False,
        config: dict[str, Any] | None = None,
        callbacks: list | None = None,
        hook_manager=None,
    ):
        self.debug = debug
        self.config = copy.deepcopy(config) if config else copy.deepcopy(DEFAULT_CONFIG)
        self.callbacks = callbacks or []
        self.hook_manager = hook_manager

        # Update the dataflows config
        set_config(self.config)

        # Create cache directory
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        llm_kwargs = self._get_provider_kwargs()
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        # Initialize 5 memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config.get("max_debate_rounds", 1),
            max_risk_discuss_rounds=self.config.get("max_risk_discuss_rounds", 1),
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100)
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # Initialize verifier if enabled
        self.data_verifier = None
        if self.config.get("verification", {}).get("enabled", False):
            try:
                from tradingagents.verification.data_verifier import DataVerifier
                self.data_verifier = DataVerifier(self.config.get("verification", {}))
            except ImportError:
                logger.warning("Verification module not available")

        # Persistence
        db_path = self.config.get("learning", {}).get("db_path", "./tradingagents_memory.db")
        self.memory_store = MemoryStore(db_path=db_path)

        # Load historical memories from SQLite into BM25
        self._load_persisted_memories()

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict: dict[str, Any] = {}

        # Compile graph
        analysts = selected_analysts or ["market", "social", "news", "fundamentals"]
        self.graph = self.graph_setup.setup_graph(analysts)

    def _load_persisted_memories(self) -> None:
        """Load historical memories from SQLite back into BM25 indexes."""
        memory_map = {
            "bull_memory": self.bull_memory,
            "bear_memory": self.bear_memory,
            "trader_memory": self.trader_memory,
            "invest_judge_memory": self.invest_judge_memory,
            "risk_manager_memory": self.risk_manager_memory,
        }
        total = 0
        for name, memory in memory_map.items():
            situations = self.memory_store.load_memories(name)
            if situations:
                memory.add_situations(situations)
                total += len(situations)

        if total > 0:
            logger.info("Loaded %d historical memories from SQLite", total)

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs: dict[str, Any] = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level
        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        return kwargs

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different analyst types."""
        return {
            "market": ToolNode([get_stock_data, get_indicators]),
            "social": ToolNode([get_news]),
            "news": ToolNode([get_news, get_global_news, get_insider_transactions]),
            "fundamentals": ToolNode([
                get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement,
            ]),
        }

    def _dispatch_hook(self, event: HookEvent, **kwargs) -> HookContext:
        """Dispatch a hook event if hook_manager is available."""
        if not self.hook_manager:
            return HookContext(event=event)

        ctx = HookContext(
            event=event,
            ticker=kwargs.get("ticker", self.ticker or ""),
            trade_date=kwargs.get("trade_date", ""),
            metadata=kwargs.get("metadata", {}),
            config=self.config,
        )
        return self.hook_manager.dispatch(ctx)

    def propagate(
        self,
        company_name: str,
        trade_date: str,
        context: dict | None = None,
    ) -> tuple[dict, str]:
        """Run the trading agents graph for a company on a specific date.

        Args:
            company_name: Ticker symbol (e.g., "NVDA")
            trade_date: Analysis date (e.g., "2026-01-15")
            context: Optional extra context (from heartbeat alerts, etc.)

        Returns:
            Tuple of (final_state dict, signal string "BUY"/"SELL"/"HOLD")
        """
        self.ticker = company_name

        # ── Hook: BEFORE_PROPAGATE ──
        before_ctx = self._dispatch_hook(
            HookEvent.BEFORE_PROPAGATE,
            ticker=company_name,
            trade_date=str(trade_date),
            metadata={"context": context or {}},
        )

        # Initialize state
        init_state = self.propagator.create_initial_state(company_name, trade_date)

        # Inject hook context (e.g., portfolio holdings)
        if before_ctx.inject_context:
            init_state["messages"].append(("system", before_ctx.inject_context))

        # Build credibility summary if verifier is available
        if self.data_verifier:
            try:
                summary = self.data_verifier.build_credibility_summary()
                init_state["data_credibility"] = summary.to_dict()
            except Exception as e:
                logger.warning(f"Credibility check failed: {e}")

        # Run graph
        args = self.propagator.get_graph_args(self.callbacks or None)

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_state, **args):
                if chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)
            final_state = trace[-1] if trace else init_state
        else:
            final_state = self.graph.invoke(init_state, **args)

        # Store state
        self.curr_state = final_state
        self._log_state(trade_date, final_state)

        # Extract signal (4-level fallback)
        decision = self.process_signal(final_state.get("final_trade_decision", ""))

        # ── Hook: AFTER_DECISION ──
        self._dispatch_hook(
            HookEvent.AFTER_DECISION,
            ticker=company_name,
            trade_date=str(trade_date),
            metadata={
                "decision": decision,
                "full_signal": final_state.get("final_trade_decision", ""),
                "trading_graph_ref": self,
            },
        )

        # ── Persist analysis result for T+N reflection ──
        try:
            consensus = final_state.get("analyst_consensus", {})
            confidence = consensus.get("confidence", "") if isinstance(consensus, dict) else ""
            self.memory_store.save_analysis_result(
                ticker=company_name,
                trade_date=str(trade_date),
                signal=decision,
                confidence=confidence,
                full_decision=final_state.get("final_trade_decision", ""),
                state_json=json.dumps(
                    self._serialize_state(final_state), default=str
                ),
            )
        except Exception:
            logger.exception("Failed to persist analysis result")

        return final_state, decision

    @staticmethod
    def _serialize_state(state: dict) -> dict:
        """Extract serializable fields from agent state for DB storage."""
        invest_debate = state.get("investment_debate_state", {})
        risk_debate = state.get("risk_debate_state", {})
        return {
            "company_of_interest": state.get("company_of_interest", ""),
            "trade_date": state.get("trade_date", ""),
            "market_report": state.get("market_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "news_report": state.get("news_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
            "analyst_consensus": state.get("analyst_consensus", {}),
            "data_credibility": state.get("data_credibility", {}),
            "investment_debate_state": {
                "bull_history": invest_debate.get("bull_history", []),
                "bear_history": invest_debate.get("bear_history", []),
                "judge_decision": invest_debate.get("judge_decision", ""),
            },
            "investment_plan": state.get("investment_plan", ""),
            "trader_investment_plan": state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "judge_decision": risk_debate.get("judge_decision", ""),
            },
            "final_trade_decision": state.get("final_trade_decision", ""),
        }

    def auto_reflect_pending(self, as_of_date: str = "") -> list[dict]:
        """Run T+N reflection on all pending analysis results.

        Returns list of reflection result dicts.
        """
        from tradingagents.learning.auto_reflect import AutoReflector

        memories = {
            "bull_memory": self.bull_memory,
            "bear_memory": self.bear_memory,
            "trader_memory": self.trader_memory,
            "invest_judge_memory": self.invest_judge_memory,
            "risk_manager_memory": self.risk_manager_memory,
        }
        horizon = self.config.get("learning", {}).get("reflection_horizon_days", 7)
        reflector = AutoReflector(
            reflector=self.reflector,
            memory_store=self.memory_store,
            memories=memories,
            horizon=horizon,
        )
        return reflector.reflect_pending(as_of_date=as_of_date)

    def _log_state(self, trade_date: str, final_state: dict) -> None:
        """Log the final state to a JSON file."""
        invest_debate = final_state.get("investment_debate_state", {})
        risk_debate = final_state.get("risk_debate_state", {})

        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state.get("company_of_interest", ""),
            "trade_date": final_state.get("trade_date", ""),
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "analyst_consensus": final_state.get("analyst_consensus", {}),
            "data_credibility": final_state.get("data_credibility", {}),
            "investment_debate_state": {
                "bull_history": invest_debate.get("bull_history", []),
                "bear_history": invest_debate.get("bear_history", []),
                "history": invest_debate.get("history", []),
                "judge_decision": invest_debate.get("judge_decision", ""),
            },
            "trader_investment_decision": final_state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "aggressive_history": risk_debate.get("aggressive_history", []),
                "conservative_history": risk_debate.get("conservative_history", []),
                "neutral_history": risk_debate.get("neutral_history", []),
                "history": risk_debate.get("history", []),
                "judge_decision": risk_debate.get("judge_decision", ""),
            },
            "investment_plan": final_state.get("investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        }

        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w") as f:
            json.dump(self.log_states_dict, f, indent=4, default=str)

    def reflect_and_remember(self, returns_losses: Any) -> None:
        """Reflect on decisions and update all 5 memories."""
        if not self.curr_state:
            logger.warning("No state to reflect on — run propagate() first")
            return

        memories = {
            "bull_memory": self.bull_memory,
            "bear_memory": self.bear_memory,
            "trader_memory": self.trader_memory,
            "invest_judge_memory": self.invest_judge_memory,
            "risk_manager_memory": self.risk_manager_memory,
        }
        reflect_memories(self.reflector, self.curr_state, returns_losses, memories)

    def process_signal(self, full_signal: str) -> str:
        """Process a signal to extract the core BUY/SELL/HOLD decision."""
        return self.signal_processor.process_signal(full_signal)
