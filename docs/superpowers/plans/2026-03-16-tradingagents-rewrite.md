# TradingAgents Rewrite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite TradingAgents core using CurrencyAgents architecture patterns, adding Hook system, Verification Layer, Notifications, and Scheduling — async from day one.

**Architecture:** CurrencyAgents-inspired multi-agent stock trading system. Factory-based analyst creation, fan-out parallel debates as LangGraph subgraphs, 4-level signal extraction, multi-source data verification, event-driven Hook system, and Cron/Heartbeat scheduling for 24/7 operation. All new modules use async; existing sync dataflows/llm_clients wrapped with `asyncio.to_thread()`.

**Tech Stack:** Python 3.10+, LangGraph, LangChain, yfinance, APScheduler, aiohttp, SQLite, rank-bm25, pytest

---

## File Structure

```
tradingagents/
├── constants.py                    # NEW: centralize node names, analyst types
├── exceptions.py                   # NEW: custom exception hierarchy
├── logging_config.py               # NEW: structured logging
├── default_config.py               # REWRITE: expanded config with hook/scheduler sections
│
├── agents/
│   ├── analysts/
│   │   ├── factory.py              # NEW: create_analyst() factory (from CurrencyAgents)
│   │   ├── market_analyst.py       # REWRITE: use factory, add credibility context
│   │   ├── social_media_analyst.py # REWRITE: use factory
│   │   ├── news_analyst.py         # REWRITE: use factory, credibility-aware prompt
│   │   └── fundamentals_analyst.py # REWRITE: use factory
│   ├── researchers/
│   │   ├── bull_researcher.py      # REWRITE: add fact-checking responsibility
│   │   └── bear_researcher.py      # REWRITE: add fact-checking responsibility
│   ├── managers/
│   │   ├── research_manager.py     # REWRITE: structured scoring
│   │   └── risk_manager.py         # REWRITE: credibility-weighted
│   ├── risk_mgmt/
│   │   ├── aggressive_debator.py   # REWRITE: use shared debate factory
│   │   ├── conservative_debator.py # REWRITE: use shared debate factory
│   │   └── neutral_debator.py      # REWRITE: use shared debate factory
│   ├── trader/
│   │   └── trader.py               # REWRITE: structured output
│   └── utils/
│       ├── agent_states.py         # REWRITE: better typing, frozen dataclasses
│       ├── memory.py               # REWRITE: add SQLite persistence layer
│       ├── agent_utils.py          # KEEP: msg_delete utility
│       ├── core_stock_tools.py     # KEEP
│       ├── technical_indicators_tools.py # KEEP
│       ├── fundamental_data_tools.py    # KEEP
│       └── news_data_tools.py      # KEEP
│
├── graph/
│   ├── trading_graph.py            # REWRITE: async-ready, hook integration
│   ├── setup.py                    # REWRITE: subgraph-based, retry policy
│   ├── invest_debate.py            # NEW: fan-out invest debate subgraph
│   ├── risk_debate.py              # NEW: fan-out risk debate subgraph
│   ├── conditional_logic.py        # REWRITE: factory-generated routers
│   ├── propagation.py              # REWRITE: enhanced state init
│   ├── reflection.py               # REWRITE: 5 reflectors, shared logic
│   ├── signal_processing.py        # REWRITE: 4-level fallback
│   ├── decision_extraction.py      # NEW: structured PurchaseDecision
│   ├── analyst_signals.py          # NEW: consensus + regime detection
│   └── prompt_utils.py             # NEW: shared prompt formatting
│
├── verification/                   # NEW: fake news defense
│   ├── __init__.py
│   ├── models.py                   # VerifiedDataPoint, NewsCredibility
│   ├── data_verifier.py            # Multi-source price cross-validation
│   └── news_verifier.py            # Source authority + pattern detection
│
├── hooks/                          # NEW: event-driven extensions
│   ├── __init__.py
│   ├── base.py                     # HookEvent, HookContext, BaseHook
│   ├── hook_manager.py             # Registry + sync/async dispatch
│   └── builtin/
│       ├── __init__.py
│       ├── journal_hook.py         # Decision logging to JSONL
│       ├── notify_hook.py          # Notification dispatch
│       ├── portfolio_hook.py       # Portfolio context injection
│       ├── ratelimit_hook.py       # API rate limiting
│       ├── memory_hook.py          # Auto-reflection trigger
│       └── integrity_hook.py       # Data integrity check
│
├── notify/                         # NEW: notification layer
│   ├── __init__.py
│   ├── base.py                     # BaseNotifier ABC
│   ├── feishu_notifier.py          # Feishu card messages
│   ├── slack_notifier.py           # Slack webhook
│   └── webhook_notifier.py         # Generic webhook
│
├── orchestrator/                   # NEW: scheduling + monitoring
│   ├── __init__.py
│   ├── scheduler.py                # APScheduler Cron jobs
│   ├── heartbeat.py                # Market anomaly monitoring
│   └── message_gateway.py          # HTTP webhook entry
│
├── learning/                       # NEW: persistence + auto-reflection
│   ├── __init__.py
│   ├── persistence.py              # SQLite MemoryStore
│   └── auto_reflect.py             # T+N automatic reflection
│
├── dataflows/                      # KEEP (minor fixes)
│   ├── interface.py                # KEEP
│   ├── y_finance.py                # KEEP
│   ├── config.py                   # FIX: remove global singleton
│   └── ...
│
└── llm_clients/                    # KEEP
    ├── base_client.py
    ├── factory.py
    ├── openai_client.py
    ├── anthropic_client.py
    ├── google_client.py
    └── validators.py

tests/
├── conftest.py                     # Shared fixtures
├── unit/
│   ├── test_constants.py
│   ├── test_exceptions.py
│   ├── test_config.py
│   ├── test_agent_states.py
│   ├── test_memory.py
│   ├── test_analyst_factory.py
│   ├── test_signal_processing.py
│   ├── test_decision_extraction.py
│   ├── test_analyst_signals.py
│   ├── test_conditional_logic.py
│   ├── test_verification.py
│   ├── test_hook_base.py
│   ├── test_hook_manager.py
│   ├── test_builtin_hooks.py
│   ├── test_notifiers.py
│   ├── test_scheduler.py
│   ├── test_heartbeat.py
│   └── test_persistence.py
└── integration/
    ├── test_graph_execution.py
    ├── test_debate_subgraphs.py
    └── test_hook_integration.py
```

---

## Chunk 1: Foundation

### Task 1: Constants and Exception Hierarchy

**Files:**
- Create: `tradingagents/constants.py`
- Create: `tradingagents/exceptions.py`
- Create: `tradingagents/logging_config.py`
- Test: `tests/unit/test_constants.py`
- Test: `tests/unit/test_exceptions.py`

- [ ] **Step 1: Create constants module**

```python
# tradingagents/constants.py
"""Centralized constants — eliminate magic strings."""

ANALYST_TYPES = ("market", "social", "news", "fundamentals")
DEFAULT_ANALYST_ORDER = list(ANALYST_TYPES)

VALID_SIGNALS = ("BUY", "SELL", "HOLD")

# Node name generators (prevent typos across files)
def analyst_node_name(analyst_type: str) -> str:
    return f"{analyst_type.capitalize()} Analyst"

def msg_clear_node_name(analyst_type: str) -> str:
    return f"Msg Clear {analyst_type.capitalize()}"

def tools_node_name(analyst_type: str) -> str:
    return f"tools_{analyst_type}"
```

- [ ] **Step 2: Create exceptions module**

```python
# tradingagents/exceptions.py
"""Custom exception hierarchy."""


class TradingAgentsError(Exception):
    """Base exception for all TradingAgents errors."""


class DataFetchError(TradingAgentsError):
    """Core data unavailable — terminates analysis."""
    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(f"Data fetch failed from {source}: {reason}")


class SignalProcessingError(TradingAgentsError):
    """Signal extraction failed after all fallback levels."""
    def __init__(self, raw_output: str):
        self.raw_output = raw_output[:200]
        super().__init__(f"Failed to extract signal from: {self.raw_output}")


class VerificationError(TradingAgentsError):
    """Data verification failed — credibility below threshold."""
    def __init__(self, source: str, confidence: float):
        self.source = source
        self.confidence = confidence
        super().__init__(f"Verification failed for {source}: confidence={confidence:.2f}")
```

- [ ] **Step 3: Create logging config**

```python
# tradingagents/logging_config.py
"""Structured logging setup."""

import logging
import sys


def setup_logging(verbosity: int = 0) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.DEBUG
    )
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    if root.handlers:
        return  # avoid duplicate handlers

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(handler)
    root.setLevel(level)

    # quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
```

- [ ] **Step 4: Write tests for constants and exceptions**

```python
# tests/unit/test_constants.py
from tradingagents.constants import (
    ANALYST_TYPES, analyst_node_name, msg_clear_node_name, tools_node_name,
)

def test_analyst_types():
    assert len(ANALYST_TYPES) == 4
    assert "market" in ANALYST_TYPES

def test_node_name_generators():
    assert analyst_node_name("market") == "Market Analyst"
    assert msg_clear_node_name("news") == "Msg Clear News"
    assert tools_node_name("social") == "tools_social"
```

```python
# tests/unit/test_exceptions.py
import pytest
from tradingagents.exceptions import (
    TradingAgentsError, DataFetchError, SignalProcessingError, VerificationError,
)

def test_data_fetch_error():
    err = DataFetchError("yfinance", "timeout")
    assert isinstance(err, TradingAgentsError)
    assert "yfinance" in str(err)

def test_signal_processing_error_truncates():
    err = SignalProcessingError("x" * 500)
    assert len(err.raw_output) == 200

def test_verification_error():
    err = VerificationError("unknown_blog", 0.3)
    assert err.confidence == 0.3
```

- [ ] **Step 5: Run tests**

Run: `cd /home/nobi/projects/TradingAgents && python -m pytest tests/unit/test_constants.py tests/unit/test_exceptions.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tradingagents/constants.py tradingagents/exceptions.py tradingagents/logging_config.py tests/unit/test_constants.py tests/unit/test_exceptions.py
git commit -m "feat: add constants, exceptions, and logging foundation"
```

---

### Task 2: Enhanced Configuration

**Files:**
- Modify: `tradingagents/default_config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write config test**

```python
# tests/unit/test_config.py
import copy
from tradingagents.default_config import DEFAULT_CONFIG

def test_config_has_required_keys():
    required = [
        "llm_provider", "deep_think_llm", "quick_think_llm",
        "max_debate_rounds", "max_risk_discuss_rounds",
        "data_vendors", "hooks", "scheduler", "heartbeat", "notify",
    ]
    for key in required:
        assert key in DEFAULT_CONFIG, f"Missing key: {key}"

def test_config_deep_copy_isolation():
    c1 = copy.deepcopy(DEFAULT_CONFIG)
    c2 = copy.deepcopy(DEFAULT_CONFIG)
    c1["hooks"]["enabled"] = False
    assert DEFAULT_CONFIG["hooks"]["enabled"] is True
    assert c2["hooks"]["enabled"] is True

def test_hooks_default_disabled():
    assert DEFAULT_CONFIG["scheduler"]["enabled"] is False
    assert DEFAULT_CONFIG["heartbeat"]["enabled"] is False

def test_data_vendors_defaults():
    vendors = DEFAULT_CONFIG["data_vendors"]
    assert vendors["core_stock_apis"] == "yfinance"
```

- [ ] **Step 2: Run test — should fail**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: FAIL (missing hooks/scheduler/heartbeat keys)

- [ ] **Step 3: Rewrite default_config.py**

```python
# tradingagents/default_config.py
"""Default configuration with all extension points."""

import os

DEFAULT_CONFIG: dict = {
    # ── Paths ──
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),

    # ── LLM ──
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.2",
    "quick_think_llm": "gpt-5-mini",
    "backend_url": "https://api.openai.com/v1",
    "google_thinking_level": None,
    "openai_reasoning_effort": None,

    # ── Graph ──
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # ── Data Vendors ──
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
    "tool_vendors": {},

    # ── Verification ──
    "verification": {
        "enabled": True,
        "price_cross_validate": True,
        "news_credibility_scoring": True,
        "min_credibility_score": 0.4,
    },

    # ── Hooks ──
    "hooks": {
        "enabled": True,
        "entries": {
            "journal": {"enabled": True, "output_dir": "./journals"},
            "notify": {"enabled": False, "channels": []},
            "portfolio_context": {"enabled": False, "portfolio_file": ""},
            "ratelimit": {"enabled": True, "max_calls_per_minute": 30},
            "auto_reflect": {"enabled": False},
            "data_integrity": {"enabled": True},
        },
    },

    # ── Scheduler ──
    "scheduler": {
        "enabled": False,
        "timezone": "US/Eastern",
        "jobs": [],
    },

    # ── Heartbeat ──
    "heartbeat": {
        "enabled": False,
        "interval_seconds": 300,
        "watchlist": [],
        "triggers": {
            "price_change_pct": 3.0,
            "volume_spike_ratio": 2.5,
            "news_keywords": [
                "earnings", "FDA", "merger", "bankruptcy",
                "lawsuit", "guidance", "upgrade", "downgrade",
            ],
        },
    },

    # ── Notify ──
    "notify": {
        "channels": [],
    },

    # ── Message Gateway ──
    "message_gateway": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 8899,
        "auth_token": "",  # Bearer token from env
    },

    # ── Learning ──
    "learning": {
        "db_path": "./tradingagents_memory.db",
        "reflection_horizon_days": 7,
    },
}
```

- [ ] **Step 4: Run test — should pass**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/default_config.py tests/unit/test_config.py
git commit -m "feat: expand config with hooks, scheduler, heartbeat, verification"
```

---

### Task 3: Enhanced Agent State Definitions

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py`
- Test: `tests/unit/test_agent_states.py`

- [ ] **Step 1: Write state tests**

```python
# tests/unit/test_agent_states.py
from tradingagents.agents.utils.agent_states import (
    AgentState, InvestDebateState, RiskDebateState,
    create_empty_invest_debate_state, create_empty_risk_debate_state,
)

def test_create_empty_invest_debate():
    state = create_empty_invest_debate_state()
    assert state["count"] == 0
    assert state["judge_decision"] == ""
    assert state["bull_history"] == []
    assert state["bear_history"] == []

def test_create_empty_risk_debate():
    state = create_empty_risk_debate_state()
    assert state["count"] == 0
    assert state["aggressive_history"] == []
    assert state["conservative_history"] == []
    assert state["neutral_history"] == []

def test_agent_state_has_required_fields():
    """Verify all fields exist in the TypedDict."""
    annotations = AgentState.__annotations__
    required = [
        "messages", "company_of_interest", "trade_date",
        "market_report", "sentiment_report", "news_report", "fundamentals_report",
        "investment_debate_state", "investment_plan",
        "trader_investment_plan", "risk_debate_state", "final_trade_decision",
        "analyst_consensus", "market_regime", "data_credibility",
    ]
    for field in required:
        assert field in annotations, f"Missing field: {field}"
```

- [ ] **Step 2: Run test — should fail**

Run: `python -m pytest tests/unit/test_agent_states.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite agent_states.py**

```python
# tradingagents/agents/utils/agent_states.py
"""Agent state definitions — single source of truth for all graph state."""

from __future__ import annotations
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


class InvestDebateState(TypedDict):
    """Bull/Bear investment debate state."""
    history: list[str]
    bull_history: list[str]
    bear_history: list[str]
    current_response: str
    judge_decision: str
    count: int


class RiskDebateState(TypedDict):
    """Aggressive/Conservative/Neutral risk debate state."""
    history: list[str]
    aggressive_history: list[str]
    conservative_history: list[str]
    neutral_history: list[str]
    latest_speaker: str
    judge_decision: str
    count: int


class AgentState(MessagesState):
    """Main graph state — all agents read/write through this."""
    company_of_interest: str
    trade_date: str
    sender: str

    # Analyst reports
    market_report: Annotated[str, "Technical analysis report"]
    sentiment_report: Annotated[str, "Social sentiment report"]
    news_report: Annotated[str, "News and macro report"]
    fundamentals_report: Annotated[str, "Financial fundamentals report"]

    # Consensus & regime (NEW)
    analyst_consensus: Annotated[dict, "Consensus voting from all analysts"]
    market_regime: Annotated[dict, "Market regime classification"]
    data_credibility: Annotated[dict, "Data credibility summary"]

    # Debate states
    investment_debate_state: InvestDebateState
    investment_plan: str
    trader_investment_plan: str
    risk_debate_state: RiskDebateState
    final_trade_decision: str


def create_empty_invest_debate_state() -> InvestDebateState:
    return InvestDebateState(
        history=[],
        bull_history=[],
        bear_history=[],
        current_response="",
        judge_decision="",
        count=0,
    )


def create_empty_risk_debate_state() -> RiskDebateState:
    return RiskDebateState(
        history=[],
        aggressive_history=[],
        conservative_history=[],
        neutral_history=[],
        latest_speaker="",
        judge_decision="",
        count=0,
    )
```

- [ ] **Step 4: Run test — should pass**

Run: `python -m pytest tests/unit/test_agent_states.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/agent_states.py tests/unit/test_agent_states.py
git commit -m "feat: enhance agent state with consensus, regime, credibility fields"
```

---

### Task 4: Fix Dataflows Global Config Singleton

**Files:**
- Modify: `tradingagents/dataflows/config.py`
- Test: `tests/unit/test_config.py` (append)

- [ ] **Step 1: Write test for thread-safe config**

Append to `tests/unit/test_config.py`:

```python
from tradingagents.dataflows.config import DataflowConfig

def test_dataflow_config_isolation():
    c1 = DataflowConfig({"data_vendors": {"core": "yfinance"}})
    c2 = DataflowConfig({"data_vendors": {"core": "alpha_vantage"}})
    assert c1.get("data_vendors")["core"] == "yfinance"
    assert c2.get("data_vendors")["core"] == "alpha_vantage"
```

- [ ] **Step 2: Rewrite config.py as instance-based**

```python
# tradingagents/dataflows/config.py
"""Instance-based configuration — no global singleton."""

from __future__ import annotations
import copy


class DataflowConfig:
    """Thread-safe, instance-scoped configuration."""

    def __init__(self, config: dict | None = None):
        self._config = copy.deepcopy(config) if config else {}

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def __getitem__(self, key: str):
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config


# Backward compatibility — will be removed in future versions
_config: dict = {}

def initialize_config(config: dict) -> None:
    global _config
    _config = copy.deepcopy(config)

def set_config(config: dict) -> None:
    global _config
    _config = copy.deepcopy(config)

def get_config() -> dict:
    return _config
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tradingagents/dataflows/config.py tests/unit/test_config.py
git commit -m "fix: replace global config singleton with instance-based DataflowConfig"
```

---

## Chunk 2: Verification Layer

### Task 5: Verification Data Models

**Files:**
- Create: `tradingagents/verification/__init__.py`
- Create: `tradingagents/verification/models.py`
- Test: `tests/unit/test_verification.py`

- [ ] **Step 1: Write test**

```python
# tests/unit/test_verification.py
from tradingagents.verification.models import (
    VerifiedDataPoint, NewsCredibility, CredibilitySummary,
)

def test_verified_data_point_single_source():
    dp = VerifiedDataPoint(value=150.0, sources=["yfinance"], confidence=0.3, discrepancies=[])
    assert dp.confidence == 0.3
    assert len(dp.sources) == 1

def test_news_credibility_defaults():
    nc = NewsCredibility(score=0.7, flags=[], source_tier="T1")
    assert nc.is_reliable

def test_news_credibility_unreliable():
    nc = NewsCredibility(score=0.3, flags=["unknown source"], source_tier="unknown")
    assert not nc.is_reliable

def test_credibility_summary_format():
    summary = CredibilitySummary(
        price_confidence=0.95,
        news_items=[
            NewsCredibility(score=0.9, flags=[], source_tier="T1"),
            NewsCredibility(score=0.3, flags=["suspicious"], source_tier="unknown"),
        ],
        warnings=["1 unverified source"],
    )
    text = summary.to_prompt_text()
    assert "DATA RELIABILITY" in text
    assert "0.95" in text
```

- [ ] **Step 2: Run test — should fail**

- [ ] **Step 3: Implement models**

```python
# tradingagents/verification/__init__.py
"""Data verification layer — multi-source validation and credibility scoring."""

# tradingagents/verification/models.py
"""Verification data models."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerifiedDataPoint:
    """A data point with cross-validation metadata."""
    value: Any
    sources: list[str]
    confidence: float  # 0.0 ~ 1.0
    discrepancies: list[str]


@dataclass(frozen=True)
class NewsCredibility:
    """Credibility assessment for a single news item."""
    score: float  # 0.0 ~ 1.0
    flags: list[str]
    source_tier: str  # "T1", "T2", "unknown"
    headline: str = ""
    source_name: str = ""

    @property
    def is_reliable(self) -> bool:
        return self.score >= 0.5


@dataclass
class CredibilitySummary:
    """Aggregated credibility summary injected into agent context."""
    price_confidence: float = 1.0
    news_items: list[NewsCredibility] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = ["=== DATA RELIABILITY ASSESSMENT ==="]
        lines.append(f"- Price data: confidence={self.price_confidence:.2f}")

        t1 = [n for n in self.news_items if n.source_tier == "T1"]
        t2 = [n for n in self.news_items if n.source_tier == "T2"]
        unk = [n for n in self.news_items if n.source_tier == "unknown"]

        if self.news_items:
            parts = []
            if t1:
                parts.append(f"{len(t1)} T1 (reliable)")
            if t2:
                parts.append(f"{len(t2)} T2 (moderate)")
            if unk:
                parts.append(f"{len(unk)} unknown (unverified)")
            lines.append(f"- News sources: {', '.join(parts)}")

        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.append("=== END ASSESSMENT ===")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "price_confidence": self.price_confidence,
            "news_reliable_count": len([n for n in self.news_items if n.is_reliable]),
            "news_unreliable_count": len([n for n in self.news_items if not n.is_reliable]),
            "warnings": self.warnings,
        }
```

- [ ] **Step 4: Run test — should pass**
- [ ] **Step 5: Commit**

```bash
git add tradingagents/verification/ tests/unit/test_verification.py
git commit -m "feat: add verification data models (VerifiedDataPoint, NewsCredibility)"
```

---

### Task 6: News Verifier

**Files:**
- Create: `tradingagents/verification/news_verifier.py`
- Test: `tests/unit/test_verification.py` (append)

- [ ] **Step 1: Append tests**

```python
# append to tests/unit/test_verification.py
from tradingagents.verification.news_verifier import NewsVerifier

class TestNewsVerifier:
    def setup_method(self):
        self.verifier = NewsVerifier()

    def test_t1_source_high_score(self):
        result = self.verifier.assess("NVDA earnings beat", "Reuters")
        assert result.source_tier == "T1"
        assert result.score >= 0.8

    def test_unknown_source_low_score(self):
        result = self.verifier.assess("NVDA to moon", "random_blog.com")
        assert result.source_tier == "unknown"
        assert result.score < 0.5

    def test_suspicious_pattern_flags(self):
        result = self.verifier.assess(
            "GUARANTEED 100% profit on NVDA insider exclusive", "Reuters"
        )
        assert len(result.flags) > 0
        assert result.score < 0.9  # penalized despite T1

    def test_t2_source_moderate(self):
        result = self.verifier.assess("Market update", "CNBC")
        assert result.source_tier == "T2"
        assert 0.5 <= result.score <= 0.9

    def test_batch_assess(self):
        items = [
            ("NVDA earnings", "Bloomberg"),
            ("Secret tip", "unknown_blog"),
        ]
        results = self.verifier.assess_batch(items)
        assert len(results) == 2
        assert results[0].is_reliable
```

- [ ] **Step 2: Implement NewsVerifier**

```python
# tradingagents/verification/news_verifier.py
"""News credibility scoring — source authority + suspicious pattern detection."""

from __future__ import annotations
import re
import logging

from .models import NewsCredibility

logger = logging.getLogger("tradingagents.verification.news")


class NewsVerifier:
    """Rule-based news credibility assessor. Zero LLM token cost."""

    T1_SOURCES = frozenset({
        "reuters", "bloomberg", "wsj", "wall street journal",
        "ft", "financial times", "sec.gov", "federalreserve.gov",
        "associated press", "ap news",
    })

    T2_SOURCES = frozenset({
        "cnbc", "marketwatch", "yahoo finance", "barrons",
        "investor's business daily", "seeking alpha", "motley fool",
        "benzinga",
    })

    SUSPICIOUS_PATTERNS = [
        (r"(?i)guaranteed.*profit", "guaranteed_profit"),
        (r"(?i)insider.*exclusive", "insider_exclusive"),
        (r"(?i)BREAKING.*100%", "exaggerated_breaking"),
        (r"(?i)secret.*tip", "secret_tip"),
        (r"(?i)act\s+now.*before", "urgency_pressure"),
        (r"(?i)sources?\s+(say|claim|reveal)", "anonymous_source"),
    ]

    def assess(
        self, headline: str, source: str, body: str = ""
    ) -> NewsCredibility:
        flags: list[str] = []
        score = 0.7  # baseline

        # Source tier classification
        source_lower = source.lower().strip()
        if any(t1 in source_lower for t1 in self.T1_SOURCES):
            source_tier = "T1"
            score += 0.2
        elif any(t2 in source_lower for t2 in self.T2_SOURCES):
            source_tier = "T2"
            score += 0.1
        else:
            source_tier = "unknown"
            score -= 0.3
            flags.append(f"Unknown source: {source}")

        # Suspicious pattern detection
        text = f"{headline} {body}"
        for pattern, label in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text):
                score -= 0.15
                flags.append(f"Suspicious: {label}")

        score = max(0.0, min(1.0, score))

        return NewsCredibility(
            score=score,
            flags=flags,
            source_tier=source_tier,
            headline=headline[:200],
            source_name=source,
        )

    def assess_batch(
        self, items: list[tuple[str, str]]
    ) -> list[NewsCredibility]:
        return [self.assess(headline, source) for headline, source in items]
```

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/verification/news_verifier.py tests/unit/test_verification.py
git commit -m "feat: add NewsVerifier with source authority + pattern detection"
```

---

### Task 7: Data Cross-Validator

**Files:**
- Create: `tradingagents/verification/data_verifier.py`
- Test: `tests/unit/test_verification.py` (append)

- [ ] **Step 1: Append tests**

```python
# append to tests/unit/test_verification.py
from tradingagents.verification.data_verifier import DataVerifier

class TestDataVerifier:
    def setup_method(self):
        self.verifier = DataVerifier()

    def test_consistent_prices_high_confidence(self):
        result = self.verifier._assess_price_consistency(
            {"source_a": 150.0, "source_b": 150.1}
        )
        assert result.confidence >= 0.9

    def test_divergent_prices_low_confidence(self):
        result = self.verifier._assess_price_consistency(
            {"source_a": 150.0, "source_b": 160.0}
        )
        assert result.confidence < 0.6
        assert len(result.discrepancies) > 0

    def test_single_source_low_confidence(self):
        result = self.verifier._assess_price_consistency(
            {"source_a": 150.0}
        )
        assert result.confidence <= 0.5
```

- [ ] **Step 2: Implement DataVerifier**

```python
# tradingagents/verification/data_verifier.py
"""Multi-source data cross-validation."""

from __future__ import annotations
import logging
from typing import Any

from .models import VerifiedDataPoint, CredibilitySummary, NewsCredibility
from .news_verifier import NewsVerifier

logger = logging.getLogger("tradingagents.verification.data")


class DataVerifier:
    """Cross-validates data from multiple sources."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.news_verifier = NewsVerifier()

    def _assess_price_consistency(
        self, prices: dict[str, float | None]
    ) -> VerifiedDataPoint:
        valid = {k: v for k, v in prices.items() if v is not None}

        if not valid:
            return VerifiedDataPoint(
                value=None, sources=[], confidence=0.0,
                discrepancies=["No price data available"],
            )

        if len(valid) == 1:
            src, val = next(iter(valid.items()))
            return VerifiedDataPoint(
                value=val, sources=[src], confidence=0.5,
                discrepancies=["Single source only"],
            )

        values = list(valid.values())
        mean_val = sum(values) / len(values)
        max_dev = max(abs(v - mean_val) / mean_val for v in values) if mean_val else 0

        discrepancies = []
        if max_dev > 0.02:
            discrepancies = [f"{k}: {v:.4f}" for k, v in valid.items()]

        confidence = 0.95 if max_dev <= 0.01 else (0.7 if max_dev <= 0.02 else 0.5)

        return VerifiedDataPoint(
            value=mean_val, sources=list(valid.keys()),
            confidence=confidence, discrepancies=discrepancies,
        )

    def assess_news(
        self, news_items: list[dict]
    ) -> list[NewsCredibility]:
        results = []
        for item in news_items:
            headline = item.get("title", item.get("headline", ""))
            source = item.get("source", item.get("publisher", "unknown"))
            cred = self.news_verifier.assess(headline, source)
            results.append(cred)
        return results

    def build_credibility_summary(
        self,
        price_data: dict[str, float | None] | None = None,
        news_items: list[dict] | None = None,
    ) -> CredibilitySummary:
        summary = CredibilitySummary()

        if price_data:
            vp = self._assess_price_consistency(price_data)
            summary.price_confidence = vp.confidence
            if vp.discrepancies:
                summary.warnings.extend(
                    [f"Price discrepancy: {d}" for d in vp.discrepancies]
                )

        if news_items:
            creds = self.assess_news(news_items)
            summary.news_items = creds
            unreliable = [c for c in creds if not c.is_reliable]
            if unreliable:
                summary.warnings.append(
                    f"{len(unreliable)} news items from unverified sources"
                )

        return summary
```

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/verification/data_verifier.py tests/unit/test_verification.py
git commit -m "feat: add DataVerifier with multi-source cross-validation"
```

---

## Chunk 3: Hook System

### Task 8: Hook Base Classes

**Files:**
- Create: `tradingagents/hooks/__init__.py`
- Create: `tradingagents/hooks/base.py`
- Test: `tests/unit/test_hook_base.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_hook_base.py
import pytest
from tradingagents.hooks.base import HookEvent, HookContext, BaseHook


def test_hook_events_are_strings():
    assert HookEvent.BEFORE_PROPAGATE == "before_propagate"
    assert HookEvent.AFTER_DECISION == "after_decision"


def test_hook_context_defaults():
    ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
    assert ctx.ticker == ""
    assert ctx.skip is False
    assert ctx.inject_context is None


def test_hook_context_mutation():
    ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE, ticker="NVDA")
    ctx.inject_context = "extra info"
    assert ctx.inject_context == "extra info"


class DummyHook(BaseHook):
    name = "dummy"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        context.metadata["touched"] = True
        return context


def test_base_hook_subclass():
    hook = DummyHook()
    assert hook.name == "dummy"
    ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
    result = hook.handle(ctx)
    assert result.metadata["touched"] is True
```

- [ ] **Step 2: Implement hook base**

```python
# tradingagents/hooks/__init__.py
"""Event-driven Hook system for TradingAgents."""

# tradingagents/hooks/base.py
"""Hook base classes — events, context, and abstract hook."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class HookEvent(str, Enum):
    """Lifecycle events hooks can subscribe to."""
    BEFORE_PROPAGATE = "before_propagate"
    AFTER_PROPAGATE = "after_propagate"
    BEFORE_ANALYST = "before_analyst"
    AFTER_ANALYST = "after_analyst"
    BEFORE_DEBATE = "before_debate"
    AFTER_DEBATE = "after_debate"
    BEFORE_DECISION = "before_decision"
    AFTER_DECISION = "after_decision"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    HEARTBEAT_TICK = "heartbeat_tick"
    HEARTBEAT_ALERT = "heartbeat_alert"
    CRON_JOB_START = "cron_job_start"
    CRON_JOB_END = "cron_job_end"
    BEFORE_REFLECT = "before_reflect"
    AFTER_REFLECT = "after_reflect"


@dataclass
class HookContext:
    """Context passed to every hook handler."""
    event: HookEvent
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str = ""
    trade_date: str = ""
    state: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    # Mutable fields — hooks can modify these to affect downstream
    inject_context: str | None = None
    skip: bool = False


class BaseHook(ABC):
    """Abstract hook — all hooks inherit this."""
    name: str = "base_hook"
    subscriptions: list[HookEvent] = []

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.enabled = True

    @abstractmethod
    def handle(self, context: HookContext) -> HookContext:
        """Process event. Return modified context."""
        return context

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} subscriptions={self.subscriptions}>"
```

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/hooks/ tests/unit/test_hook_base.py
git commit -m "feat: add Hook base classes (HookEvent, HookContext, BaseHook)"
```

---

### Task 9: Hook Manager

**Files:**
- Create: `tradingagents/hooks/hook_manager.py`
- Test: `tests/unit/test_hook_manager.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_hook_manager.py
from tradingagents.hooks.base import BaseHook, HookEvent, HookContext
from tradingagents.hooks.hook_manager import HookManager


class CountingHook(BaseHook):
    name = "counter"
    subscriptions = [HookEvent.AFTER_DECISION]

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def handle(self, context: HookContext) -> HookContext:
        self.call_count += 1
        return context


class SkippingHook(BaseHook):
    name = "skipper"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        context.skip = True
        return context


class FailingHook(BaseHook):
    name = "failer"
    subscriptions = [HookEvent.AFTER_DECISION]

    def handle(self, context: HookContext) -> HookContext:
        raise RuntimeError("hook error")


def test_register_and_dispatch():
    mgr = HookManager()
    hook = CountingHook()
    mgr.register(hook)

    ctx = HookContext(event=HookEvent.AFTER_DECISION)
    mgr.dispatch(ctx)
    assert hook.call_count == 1


def test_skip_stops_chain():
    mgr = HookManager()
    skipper = SkippingHook()
    counter = CountingHook()
    counter.subscriptions = [HookEvent.BEFORE_PROPAGATE]

    mgr.register(skipper)
    mgr.register(counter)

    ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
    result = mgr.dispatch(ctx)
    assert result.skip is True
    assert counter.call_count == 0  # skipped


def test_failing_hook_does_not_crash():
    mgr = HookManager()
    mgr.register(FailingHook())
    counter = CountingHook()
    mgr.register(counter)

    ctx = HookContext(event=HookEvent.AFTER_DECISION)
    result = mgr.dispatch(ctx)  # should not raise
    assert counter.call_count == 1  # next hook still runs


def test_unregister():
    mgr = HookManager()
    hook = CountingHook()
    mgr.register(hook)
    mgr.unregister("counter")
    assert mgr.summary["total"] == 0


def test_dispatch_unsubscribed_event():
    mgr = HookManager()
    hook = CountingHook()
    mgr.register(hook)

    ctx = HookContext(event=HookEvent.BEFORE_PROPAGATE)
    mgr.dispatch(ctx)
    assert hook.call_count == 0  # not subscribed
```

- [ ] **Step 2: Implement HookManager (synchronous)**

```python
# tradingagents/hooks/hook_manager.py
"""Hook manager — central registry and synchronous event dispatcher."""

from __future__ import annotations
import logging
from collections import defaultdict

from .base import BaseHook, HookEvent, HookContext

logger = logging.getLogger("tradingagents.hooks")


class HookManager:
    """Manages hook registration and event dispatch.

    Design: synchronous dispatch to avoid async/sync bridging issues.
    Hooks that need I/O (notifications) should use threading internally.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._hooks: dict[HookEvent, list[BaseHook]] = defaultdict(list)
        self._all_hooks: list[BaseHook] = []

    def register(self, hook: BaseHook) -> None:
        self._all_hooks.append(hook)
        for event in hook.subscriptions:
            self._hooks[event].append(hook)
            logger.info(f"Registered hook: {hook.name} -> {event.value}")

    def unregister(self, hook_name: str) -> None:
        self._all_hooks = [h for h in self._all_hooks if h.name != hook_name]
        for event in self._hooks:
            self._hooks[event] = [
                h for h in self._hooks[event] if h.name != hook_name
            ]

    def dispatch(self, context: HookContext) -> HookContext:
        """Synchronous dispatch — hooks run in registration order."""
        hooks = self._hooks.get(context.event, [])

        for hook in hooks:
            if not hook.enabled:
                continue
            try:
                context = hook.handle(context)
                if context.skip:
                    logger.info(
                        f"Hook {hook.name} set skip=True for {context.event.value}"
                    )
                    break
            except Exception as e:
                logger.error(f"Hook {hook.name} failed on {context.event.value}: {e}")

        return context

    def load_builtin_hooks(self) -> None:
        """Load enabled builtin hooks from config."""
        hook_configs = self.config.get("hooks", {}).get("entries", {})

        if hook_configs.get("journal", {}).get("enabled", False):
            from .builtin.journal_hook import JournalHook
            self.register(JournalHook(hook_configs.get("journal", {})))

        if hook_configs.get("notify", {}).get("enabled", False):
            from .builtin.notify_hook import NotifyHook
            self.register(NotifyHook(
                hook_configs.get("notify", {}),
                notify_config=self.config.get("notify", {}),
            ))

        if hook_configs.get("portfolio_context", {}).get("enabled", False):
            from .builtin.portfolio_hook import PortfolioContextHook
            self.register(PortfolioContextHook(
                hook_configs.get("portfolio_context", {})
            ))

        if hook_configs.get("ratelimit", {}).get("enabled", False):
            from .builtin.ratelimit_hook import RateLimitHook
            self.register(RateLimitHook(hook_configs.get("ratelimit", {})))

        if hook_configs.get("data_integrity", {}).get("enabled", False):
            from .builtin.integrity_hook import DataIntegrityHook
            self.register(DataIntegrityHook(hook_configs.get("data_integrity", {})))

        logger.info(f"Loaded {len(self._all_hooks)} builtin hooks")

    @property
    def summary(self) -> dict:
        return {
            "total": len(self._all_hooks),
            "hooks": [
                {
                    "name": h.name,
                    "enabled": h.enabled,
                    "events": [e.value for e in h.subscriptions],
                }
                for h in self._all_hooks
            ],
        }
```

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/hooks/hook_manager.py tests/unit/test_hook_manager.py
git commit -m "feat: add synchronous HookManager with chain dispatch"
```

---

### Task 10: Builtin Hooks

**Files:**
- Create: `tradingagents/hooks/builtin/__init__.py`
- Create: `tradingagents/hooks/builtin/journal_hook.py`
- Create: `tradingagents/hooks/builtin/ratelimit_hook.py`
- Create: `tradingagents/hooks/builtin/portfolio_hook.py`
- Create: `tradingagents/hooks/builtin/integrity_hook.py`
- Create: `tradingagents/hooks/builtin/notify_hook.py`
- Create: `tradingagents/hooks/builtin/memory_hook.py`
- Test: `tests/unit/test_builtin_hooks.py`

- [ ] **Step 1: Write tests for journal and ratelimit hooks**

```python
# tests/unit/test_builtin_hooks.py
import json
import tempfile
from pathlib import Path

from tradingagents.hooks.base import HookEvent, HookContext
from tradingagents.hooks.builtin.journal_hook import JournalHook
from tradingagents.hooks.builtin.ratelimit_hook import RateLimitHook
from tradingagents.hooks.builtin.integrity_hook import DataIntegrityHook


class TestJournalHook:
    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = JournalHook({"output_dir": tmpdir})
            ctx = HookContext(
                event=HookEvent.AFTER_DECISION,
                ticker="NVDA",
                trade_date="2026-01-15",
                metadata={"decision": "BUY", "full_signal": "Strong buy signal"},
            )
            hook.handle(ctx)

            files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(files) == 1
            with open(files[0]) as f:
                entry = json.loads(f.readline())
            assert entry["ticker"] == "NVDA"
            assert entry["decision"] == "BUY"


class TestRateLimitHook:
    def test_allows_under_limit(self):
        hook = RateLimitHook({"max_calls_per_minute": 100})
        ctx = HookContext(
            event=HookEvent.BEFORE_TOOL_CALL,
            metadata={"tool_name": "get_stock_data"},
        )
        result = hook.handle(ctx)
        assert "tool_call_start_time" in result.metadata

    def test_tracks_stats(self):
        hook = RateLimitHook({"max_calls_per_minute": 100})
        ctx = HookContext(
            event=HookEvent.AFTER_TOOL_CALL,
            metadata={"tool_name": "get_stock_data", "tool_call_start_time": 1000.0},
        )
        hook.handle(ctx)
        assert hook.stats_summary["get_stock_data"]["count"] == 1


class TestDataIntegrityHook:
    def test_flags_low_credibility(self):
        hook = DataIntegrityHook({})
        ctx = HookContext(
            event=HookEvent.AFTER_ANALYST,
            metadata={
                "data_credibility": {
                    "news_unreliable_count": 5,
                    "warnings": ["5 unverified sources"],
                }
            },
        )
        result = hook.handle(ctx)
        assert result.inject_context is not None
        assert "INTEGRITY WARNING" in result.inject_context
```

- [ ] **Step 2: Implement all builtin hooks**

Implement all 6 builtin hooks following the patterns established in the upgrade plan (Section 4.2.3). Key design decisions:
- All hooks are **synchronous** (`def handle`, not `async def handle`)
- NotifyHook uses `threading.Thread` for async I/O internally
- RateLimitHook uses `time.monotonic()` for timing (not `time.time()`)
- JournalHook writes JSONL organized by ticker + date
- IntegrityHook checks `data_credibility` metadata from VerificationLayer

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/hooks/builtin/ tests/unit/test_builtin_hooks.py
git commit -m "feat: add builtin hooks (journal, ratelimit, portfolio, integrity, notify, memory)"
```

---

## Chunk 4: Notification Layer

### Task 11: Notifier Implementations

**Files:**
- Create: `tradingagents/notify/__init__.py`
- Create: `tradingagents/notify/base.py`
- Create: `tradingagents/notify/feishu_notifier.py`
- Create: `tradingagents/notify/slack_notifier.py`
- Create: `tradingagents/notify/webhook_notifier.py`
- Test: `tests/unit/test_notifiers.py`

- [ ] **Step 1: Write tests (mock HTTP)**

```python
# tests/unit/test_notifiers.py
from unittest.mock import patch, AsyncMock
import pytest

from tradingagents.notify.base import BaseNotifier
from tradingagents.notify.feishu_notifier import FeishuNotifier
from tradingagents.notify.slack_notifier import SlackNotifier
from tradingagents.notify.webhook_notifier import WebhookNotifier


def test_feishu_card_message_format():
    notifier = FeishuNotifier({
        "webhook_url": "https://example.com/hook",
        "msg_type": "interactive",
    })
    payload = notifier._build_card_message("🟢 **NVDA** — BUY\nDate: 2026-01-15")
    assert payload["msg_type"] == "interactive"
    assert payload["card"]["header"]["template"] == "green"


def test_feishu_sell_card_is_red():
    notifier = FeishuNotifier({"webhook_url": "https://example.com/hook"})
    payload = notifier._build_card_message("🔴 **NVDA** — SELL")
    assert payload["card"]["header"]["template"] == "red"


def test_feishu_sign_generation():
    sign = FeishuNotifier._gen_sign("1234567890", "secret123")
    assert isinstance(sign, str)
    assert len(sign) > 0


def test_notifier_returns_false_without_url():
    notifier = SlackNotifier({"webhook_url": ""})
    # send() should return False for empty URL
    # (actual HTTP call mocked in integration tests)
```

- [ ] **Step 2: Implement notifiers**

Implement all 3 notifiers following the patterns from Section 4.6 of the upgrade plan. Use `requests` (sync) instead of `aiohttp` for simplicity since NotifyHook dispatches via threading.

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/notify/ tests/unit/test_notifiers.py
git commit -m "feat: add notification layer (Feishu, Slack, Webhook)"
```

---

## Chunk 5: Agent Factory + Analyst Rewrites

### Task 12: Analyst Factory Pattern

**Files:**
- Create: `tradingagents/agents/analysts/factory.py`
- Test: `tests/unit/test_analyst_factory.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_analyst_factory.py
from unittest.mock import MagicMock, patch
from tradingagents.agents.analysts.factory import create_analyst


def test_factory_returns_callable():
    mock_llm = MagicMock()
    tools = []
    node_fn = create_analyst(
        llm=mock_llm,
        tools=tools,
        system_message="Analyze the market.",
        output_key="market_report",
    )
    assert callable(node_fn)


def test_factory_output_key():
    """Factory should write report to the specified output key."""
    mock_llm = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Analysis report"
    mock_result.tool_calls = []

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result
    mock_llm.bind_tools.return_value = mock_chain

    # Patch ChatPromptTemplate to return a simple passthrough
    with patch("tradingagents.agents.analysts.factory.ChatPromptTemplate") as mock_prompt:
        mock_prompt.from_messages.return_value = mock_chain
        # Re-mock the pipe
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        node_fn = create_analyst(
            llm=mock_llm, tools=[], system_message="Test", output_key="test_report",
        )
        # This is a simplified test — full integration tested separately
        assert node_fn is not None
```

- [ ] **Step 2: Implement factory**

```python
# tradingagents/agents/analysts/factory.py
"""Analyst factory — DRY creation of analyst node functions."""

from __future__ import annotations
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger("tradingagents.agents.analysts")


def create_analyst(
    llm: Any,
    tools: list,
    system_message: str,
    output_key: str,
    credibility_aware: bool = False,
) -> callable:
    """Create an analyst node function for the LangGraph.

    Args:
        llm: LangChain-compatible LLM instance
        tools: List of tools to bind
        system_message: System prompt for this analyst
        output_key: State key to write the report to
        credibility_aware: If True, append credibility context to prompt

    Returns:
        Node function: (state) -> dict
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm.bind_tools(tools)

    def analyst_node(state: dict) -> dict:
        messages = state.get("messages", [])

        # Inject credibility context if available
        if credibility_aware and state.get("data_credibility"):
            cred = state["data_credibility"]
            if cred.get("warnings"):
                cred_text = "\n".join(
                    [f"⚠️ {w}" for w in cred["warnings"]]
                )
                messages = [("system", f"DATA WARNINGS:\n{cred_text}")] + list(messages)

        result = chain.invoke({"messages": messages})

        report = result.content if hasattr(result, "content") else str(result)

        return {
            output_key: report,
            "messages": [result],
        }

    return analyst_node
```

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Rewrite 4 analyst modules to use factory**

Rewrite each analyst file to be a thin wrapper calling `create_analyst()` with analyst-specific system prompt and tools. Each file should be <30 lines.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/ tests/unit/test_analyst_factory.py
git commit -m "feat: add analyst factory, rewrite 4 analysts to use it"
```

---

### Task 13: Rewrite Researchers with Fact-Checking

**Files:**
- Modify: `tradingagents/agents/researchers/bull_researcher.py`
- Modify: `tradingagents/agents/researchers/bear_researcher.py`

- [ ] **Step 1: Rewrite bull_researcher.py**

Add fact-checking responsibility to the prompt. When counter-arguing, explicitly require challenging unverified data sources. Include memory retrieval.

- [ ] **Step 2: Rewrite bear_researcher.py**

Mirror of bull, with bearish stance + explicit instruction to flag single-source claims.

- [ ] **Step 3: Commit**

```bash
git add tradingagents/agents/researchers/
git commit -m "feat: rewrite researchers with fact-checking responsibility"
```

---

### Task 14: Rewrite Risk Debaters + Managers + Trader

**Files:**
- Modify: `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/conservative_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/neutral_debator.py`
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/managers/risk_manager.py`
- Modify: `tradingagents/agents/trader/trader.py`

- [ ] **Step 1: Rewrite risk debaters — reduce copy-paste via shared helper**

Create a `_create_risk_debator()` helper that parameterizes the stance (aggressive/conservative/neutral) with specific prompt sections.

- [ ] **Step 2: Rewrite research_manager with structured scoring**

Add 3-dimensional scoring (evidence quality, risk-reward, timing) following CurrencyAgents' invest judge pattern. Include credibility-weighted evaluation.

- [ ] **Step 3: Rewrite risk_manager with credibility awareness**

Risk judge receives data credibility summary and adjusts position sizing downward when credibility is low.

- [ ] **Step 4: Rewrite trader with structured output**

Trader generates concrete execution parameters: position %, timeline, stop-loss.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/risk_mgmt/ tradingagents/agents/managers/ tradingagents/agents/trader/
git commit -m "feat: rewrite debaters, managers, trader with structured scoring + credibility"
```

---

## Chunk 6: Graph Rewrite

### Task 15: Signal Processing (4-Level Fallback)

**Files:**
- Rewrite: `tradingagents/graph/signal_processing.py`
- Test: `tests/unit/test_signal_processing.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_signal_processing.py
import pytest
from tradingagents.graph.signal_processing import extract_signal, SignalProcessor


class TestExtractSignal:
    def test_buy_regex(self):
        assert extract_signal("I recommend BUY for this stock") == "BUY"

    def test_sell_regex(self):
        assert extract_signal("Final decision: SELL") == "SELL"

    def test_hold_regex(self):
        assert extract_signal("HOLD position") == "HOLD"

    def test_case_insensitive(self):
        assert extract_signal("buy this") == "BUY"

    def test_returns_none_on_garbage(self):
        assert extract_signal("no signal here") is None

    def test_last_occurrence_wins(self):
        assert extract_signal("BUY but then SELL") == "SELL"


class TestSignalProcessor:
    def test_direct_extraction(self):
        processor = SignalProcessor(llm=None)
        result = processor.process_signal("Final: BUY")
        assert result == "BUY"

    def test_raises_on_empty(self):
        processor = SignalProcessor(llm=None)
        with pytest.raises(Exception):
            processor.process_signal("")
```

- [ ] **Step 2: Implement 4-level signal processing**

```python
# tradingagents/graph/signal_processing.py
"""4-level signal extraction — regex → LLM → strict LLM → error."""

from __future__ import annotations
import re
import logging

from tradingagents.exceptions import SignalProcessingError

logger = logging.getLogger("tradingagents.graph.signal")

_SIGNAL_RE = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)


def extract_signal(text: str) -> str | None:
    """Extract trading signal via regex. Returns last match (uppercase) or None."""
    matches = _SIGNAL_RE.findall(text)
    return matches[-1].upper() if matches else None


class SignalProcessor:
    """Progressive fallback signal extraction."""

    def __init__(self, llm=None):
        self.llm = llm

    def process_signal(self, full_signal: str) -> str:
        if not full_signal or not full_signal.strip():
            raise SignalProcessingError(full_signal or "(empty)")

        # Level 1: Direct regex
        result = extract_signal(full_signal)
        if result:
            logger.debug(f"Signal extracted (L1 regex): {result}")
            return result

        # Level 2: LLM extraction
        if self.llm:
            try:
                llm_out = self.llm.invoke([
                    ("system", "Extract the trading signal (BUY, SELL, or HOLD) from the text below. Output ONLY the signal word."),
                    ("human", full_signal[:2000]),
                ]).content
                result = extract_signal(llm_out)
                if result:
                    logger.debug(f"Signal extracted (L2 LLM): {result}")
                    return result
            except Exception as e:
                logger.warning(f"L2 LLM extraction failed: {e}")

        # Level 3: Strict LLM
        if self.llm:
            try:
                llm_out = self.llm.invoke([
                    ("system", "Output EXACTLY one word: BUY or SELL or HOLD. Nothing else."),
                    ("human", f"What is the trading signal?\n\n{full_signal[:1000]}"),
                ]).content.strip().upper()
                if llm_out in ("BUY", "SELL", "HOLD"):
                    logger.debug(f"Signal extracted (L3 strict): {llm_out}")
                    return llm_out
            except Exception as e:
                logger.warning(f"L3 strict extraction failed: {e}")

        # Level 4: Error
        raise SignalProcessingError(full_signal)
```

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/graph/signal_processing.py tests/unit/test_signal_processing.py
git commit -m "feat: rewrite signal processing with 4-level fallback"
```

---

### Task 16: Analyst Signals (Consensus + Regime Detection)

**Files:**
- Create: `tradingagents/graph/analyst_signals.py`
- Test: `tests/unit/test_analyst_signals.py`

- [ ] **Step 1: Write tests**

Test `extract_direction()`, `compute_consensus()`, `detect_regime()`. Follow CurrencyAgents patterns — consensus counts BUY/SELL/HOLD votes from 4 analysts, determines HIGH/MEDIUM/LOW confidence.

- [ ] **Step 2: Implement analyst_signals.py**

Port from CurrencyAgents: direction extraction (regex first, fallback to signal processor), consensus computation, Hurst exponent regime detection.

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/graph/analyst_signals.py tests/unit/test_analyst_signals.py
git commit -m "feat: add analyst consensus computation + regime detection"
```

---

### Task 17: Invest Debate Subgraph (Fan-Out)

**Files:**
- Create: `tradingagents/graph/invest_debate.py`
- Create: `tradingagents/graph/prompt_utils.py`
- Test: `tests/integration/test_debate_subgraphs.py`

- [ ] **Step 1: Implement invest_debate.py**

Port CurrencyAgents' invest_debate pattern:
- Fan-out parallel: Bull + Bear execute simultaneously per round
- Merge round: concatenate outputs, increment count
- Judge node: 3-dimensional scoring with credibility weighting
- Memory retrieval integrated into debater prompts

- [ ] **Step 2: Implement prompt_utils.py**

Shared formatters: `format_consensus()`, `format_regime()`, `format_credibility()`.

- [ ] **Step 3: Write integration test with mock LLM**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/graph/invest_debate.py tradingagents/graph/prompt_utils.py tests/integration/test_debate_subgraphs.py
git commit -m "feat: add fan-out invest debate subgraph with structured scoring"
```

---

### Task 18: Risk Debate Subgraph (Fan-Out)

**Files:**
- Create: `tradingagents/graph/risk_debate.py`

- [ ] **Step 1: Implement risk_debate.py**

Port CurrencyAgents' risk_debate pattern:
- Fan-out parallel: Aggressive + Conservative + Neutral per round
- Risk judge with 3-dimensional scoring
- Credibility-weighted position sizing

- [ ] **Step 2: Append integration test**
- [ ] **Step 3: Commit**

```bash
git add tradingagents/graph/risk_debate.py tests/integration/test_debate_subgraphs.py
git commit -m "feat: add fan-out risk debate subgraph"
```

---

### Task 19: Decision Extraction (Structured Output)

**Files:**
- Create: `tradingagents/graph/decision_extraction.py`
- Test: `tests/unit/test_decision_extraction.py`

- [ ] **Step 1: Write tests**

Test JSON extraction, validation, confidence override, low-confidence downgrade.

- [ ] **Step 2: Implement decision_extraction.py**

Port CurrencyAgents pattern: `TradingDecision` dataclass with signal, confidence, position_pct, execution_weeks, stop_loss_pct. LLM-based extraction with regex fallback. Confidence override from consensus.

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/graph/decision_extraction.py tests/unit/test_decision_extraction.py
git commit -m "feat: add structured decision extraction with validation"
```

---

### Task 20: Graph Setup + Conditional Logic Rewrite

**Files:**
- Rewrite: `tradingagents/graph/setup.py`
- Rewrite: `tradingagents/graph/conditional_logic.py`
- Test: `tests/unit/test_conditional_logic.py`

- [ ] **Step 1: Rewrite conditional_logic.py**

Factory-generated routing: `make_analyst_router(analyst_type)` returns a closure that checks tool_calls. Use constants for node names.

- [ ] **Step 2: Rewrite setup.py**

Use registry pattern for analysts. Wire invest_debate and risk_debate as subgraphs. Add NODE_RETRY policy. Integrate `analyst_signals` node between analysts and debate.

- [ ] **Step 3: Write tests for conditional logic**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/graph/setup.py tradingagents/graph/conditional_logic.py tests/unit/test_conditional_logic.py
git commit -m "feat: rewrite graph setup with subgraphs, retry policy, factory routing"
```

---

### Task 21: Trading Graph Orchestrator + Propagation

**Files:**
- Rewrite: `tradingagents/graph/trading_graph.py`
- Rewrite: `tradingagents/graph/propagation.py`
- Rewrite: `tradingagents/graph/reflection.py`

- [ ] **Step 1: Rewrite propagation.py**

Enhanced state initialization with empty consensus/regime/credibility fields. Port `get_graph_args()` with RetryPolicy.

- [ ] **Step 2: Rewrite reflection.py**

Port CurrencyAgents' 5-reflector pattern with `reflect_memories()` shared function.

- [ ] **Step 3: Rewrite trading_graph.py**

Major rewrite:
- Accept `hook_manager` in constructor
- `propagate()`: dispatch BEFORE_PROPAGATE hook → init state → build credibility summary (DataVerifier) → inject credibility + portfolio context → run graph → extract signal (4-level) → extract decision (structured) → dispatch AFTER_DECISION hook → return
- `reflect_and_remember()`: delegate to reflector + persist to SQLite
- Remove wildcard imports, use explicit imports from constants

- [ ] **Step 4: Write integration test for full graph execution with mock LLM**
- [ ] **Step 5: Commit**

```bash
git add tradingagents/graph/trading_graph.py tradingagents/graph/propagation.py tradingagents/graph/reflection.py tests/integration/test_graph_execution.py
git commit -m "feat: rewrite trading graph orchestrator with hooks + verification"
```

---

## Chunk 7: Learning & Persistence

### Task 22: SQLite Memory Store

**Files:**
- Create: `tradingagents/learning/__init__.py`
- Create: `tradingagents/learning/persistence.py`
- Test: `tests/unit/test_persistence.py`

- [ ] **Step 1: Write tests**

Test: create DB, save/load memories, save analysis results, get pending reflections, idempotent writes.

- [ ] **Step 2: Implement persistence.py**

Port CurrencyAgents' MemoryStore: 3 tables (memories, analysis_results, timing), parameterized SQL, content-hash deduplication, file permissions 0o600.

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/learning/ tests/unit/test_persistence.py
git commit -m "feat: add SQLite MemoryStore for persistent learning"
```

---

### Task 23: Enhanced BM25 Memory with Persistence

**Files:**
- Modify: `tradingagents/agents/utils/memory.py`
- Test: `tests/unit/test_memory.py`

- [ ] **Step 1: Write tests**

Test: add situations, get memories, save/load from SQLite, empty memory returns [].

- [ ] **Step 2: Enhance memory.py**

Add `save_to_store()` and `load_from_store()` methods that delegate to MemoryStore. Keep BM25 core unchanged.

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/utils/memory.py tests/unit/test_memory.py
git commit -m "feat: enhance BM25 memory with SQLite persistence"
```

---

## Chunk 8: Orchestrator (Scheduler + Heartbeat + Gateway)

### Task 24: Cron Scheduler

**Files:**
- Create: `tradingagents/orchestrator/__init__.py`
- Create: `tradingagents/orchestrator/scheduler.py`
- Test: `tests/unit/test_scheduler.py`

- [ ] **Step 1: Write tests**

Test: job creation, cron parsing, job listing, status reporting. Mock APScheduler internals.

- [ ] **Step 2: Implement scheduler.py**

Port from upgrade plan Section 4.3. Use APScheduler's `AsyncIOScheduler`. Add retry with exponential backoff. Dispatch CRON_JOB_START/END hooks.

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/orchestrator/ tests/unit/test_scheduler.py
git commit -m "feat: add APScheduler-based cron scheduler"
```

---

### Task 25: Market Heartbeat

**Files:**
- Create: `tradingagents/orchestrator/heartbeat.py`
- Test: `tests/unit/test_heartbeat.py`

- [ ] **Step 1: Write tests**

Test: baseline initialization, price spike detection, volume spike detection, cooldown mechanism. Mock yfinance.

- [ ] **Step 2: Implement heartbeat.py**

Port from upgrade plan Section 4.4, with additions:
- `exchange_calendars` for trading day checks
- Cooldown dict: `{ticker: last_alert_time}` prevents re-alerting within N minutes
- `pending_events` accumulation

- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git add tradingagents/orchestrator/heartbeat.py tests/unit/test_heartbeat.py
git commit -m "feat: add market heartbeat monitor with cooldown"
```

---

### Task 26: Message Gateway (with Auth)

**Files:**
- Create: `tradingagents/orchestrator/message_gateway.py`

- [ ] **Step 1: Implement message_gateway.py**

Port from upgrade plan Section 4.7, with added Bearer token authentication middleware. Token read from config `message_gateway.auth_token` or env `TRADINGAGENTS_GATEWAY_TOKEN`.

- [ ] **Step 2: Commit**

```bash
git add tradingagents/orchestrator/message_gateway.py
git commit -m "feat: add HTTP message gateway with Bearer auth"
```

---

## Chunk 9: Integration + Entry Point

### Task 27: Enhanced Entry Point

**Files:**
- Create: `main_enhanced.py`

- [ ] **Step 1: Implement main_enhanced.py**

Port from upgrade plan Section 4.9. Use `copy.deepcopy(DEFAULT_CONFIG)`. Startup sequence: hooks → graph → heartbeat → scheduler → gateway. Graceful shutdown with signal handlers.

- [ ] **Step 2: Commit**

```bash
git add main_enhanced.py
git commit -m "feat: add enhanced entry point with full orchestration stack"
```

---

### Task 28: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new dependencies**

```
apscheduler>=3.10
aiohttp>=3.9
exchange-calendars>=4.5
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add apscheduler, aiohttp, exchange-calendars deps"
```

---

### Task 29: Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`

- [ ] **Step 1: Create conftest with shared fixtures**

```python
# tests/conftest.py
"""Shared test fixtures."""
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """Mock LLM that returns configurable responses."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "BUY"
    response.tool_calls = []
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def sample_state():
    """Minimal AgentState for testing."""
    from tradingagents.agents.utils.agent_states import (
        create_empty_invest_debate_state,
        create_empty_risk_debate_state,
    )
    return {
        "messages": [],
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "sender": "",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "analyst_consensus": {},
        "market_regime": {},
        "data_credibility": {},
        "investment_debate_state": create_empty_invest_debate_state(),
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": create_empty_risk_debate_state(),
        "final_trade_decision": "",
    }
```

- [ ] **Step 2: Commit**

```bash
git add tests/
git commit -m "test: add test infrastructure with shared fixtures"
```

---

### Task 30: Full Integration Test

**Files:**
- Create: `tests/integration/test_hook_integration.py`

- [ ] **Step 1: Write end-to-end hook integration test**

Test: create HookManager → register journal + integrity hooks → dispatch BEFORE_PROPAGATE → verify context injection → dispatch AFTER_DECISION → verify JSONL written.

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_hook_integration.py
git commit -m "test: add end-to-end hook integration test"
```

---

## Execution Notes

### Dependencies Between Tasks

```
Task 1 (constants/exceptions) ─→ ALL subsequent tasks
Task 2 (config) ─→ Tasks 8-11, 24-27
Task 3 (agent states) ─→ Tasks 12-21
Task 4 (dataflow config) ─→ Tasks 20-21
Tasks 5-7 (verification) ─→ Tasks 12-14, 20-21
Tasks 8-10 (hooks) ─→ Tasks 20-21, 24-27
Task 11 (notify) ─→ Task 10
Tasks 12-14 (agents) ─→ Tasks 17-21
Task 15 (signal) ─→ Tasks 17-21
Task 16 (consensus) ─→ Tasks 17-21
Tasks 17-18 (debates) ─→ Task 20
Task 19 (decision) ─→ Task 21
Tasks 20-21 (graph) ─→ Tasks 27, 30
Tasks 22-23 (persistence) ─→ Task 21
Tasks 24-26 (orchestrator) ─→ Task 27
```

### Parallelizable Groups

These task groups can be developed in parallel by independent agents:

- **Group A (Foundation):** Tasks 1-4 (sequential within group)
- **Group B (Verification):** Tasks 5-7 (after Task 1)
- **Group C (Hooks):** Tasks 8-10 (after Task 1-2)
- **Group D (Notify):** Task 11 (after Task 1)
- **Group E (Agents):** Tasks 12-14 (after Tasks 1, 3)
- **Group F (Signal+Consensus):** Tasks 15-16 (after Task 1)
- **Group G (Persistence):** Tasks 22-23 (after Task 1)
- **Group H (Orchestrator):** Tasks 24-26 (after Tasks 1-2, 8)

After all groups complete:
- **Group I (Graph Assembly):** Tasks 17-21 (needs A-F, G)
- **Group J (Integration):** Tasks 27-30 (needs everything)
