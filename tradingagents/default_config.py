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
        "auth_token": "",
    },

    # ── Learning ──
    "learning": {
        "db_path": "./tradingagents_memory.db",
        "reflection_horizon_days": 7,
    },
}
