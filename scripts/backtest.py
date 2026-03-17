#!/usr/bin/env python3
"""Simple backtest script — runs propagate() over a date range.

Usage:
    python scripts/backtest.py --tickers NVDA,AAPL --start 2024-11-01 --end 2025-03-14 --interval weekly

Environment:
    Same as main_enhanced.py (OPENAI_API_KEY, FEISHU_WEBHOOK_URL, etc.)
    Feishu notifications are DISABLED during backtest to avoid spam.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.learning.persistence import MemoryStore
from tradingagents.logging_config import setup_logging

setup_logging(verbosity=1)
logger = logging.getLogger("tradingagents.backtest")


def generate_dates(
    start: str, end: str, interval: str = "weekly"
) -> list[str]:
    """Generate trading dates (skip weekends)."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    step = {"daily": 1, "weekly": 7, "biweekly": 14}[interval]
    dates: list[str] = []
    current = start_dt

    while current <= end_dt:
        # Skip weekends
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=step)

    return dates


def build_backtest_config() -> dict:
    """Config with notifications disabled."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["llm_provider"] = "openai"
    config["deep_think_llm"] = "gpt-4o-mini"
    config["quick_think_llm"] = "gpt-4o-mini"

    # Enable Feishu notifications during backtest
    feishu_url = os.environ.get("FEISHU_WEBHOOK_URL", "")
    if feishu_url:
        feishu_cfg: dict = {"webhook_url": feishu_url}
        feishu_secret = os.environ.get("FEISHU_WEBHOOK_SECRET", "")
        if feishu_secret:
            feishu_cfg["secret"] = feishu_secret
        config["hooks"]["entries"]["notify"] = {
            "enabled": True,
            "notifier": "feishu",
            "notifier_config": feishu_cfg,
        }
    config["hooks"]["entries"]["journal"]["enabled"] = True
    config["hooks"]["entries"]["ratelimit"]["enabled"] = True
    config["hooks"]["entries"]["data_integrity"]["enabled"] = True

    return config


def run_backtest(
    tickers: list[str],
    dates: list[str],
    config: dict,
) -> list[dict]:
    """Run propagate() for each ticker × date combination."""
    from tradingagents.hooks.hook_manager import HookManager

    hook_manager = HookManager(config)
    hook_manager.load_builtin_hooks()

    ta = TradingAgentsGraph(
        debug=False,
        config=config,
        hook_manager=hook_manager,
    )

    store = ta.memory_store
    total = len(tickers) * len(dates)
    results: list[dict] = []

    # Check which (ticker, date) pairs are already done
    existing: set[tuple[str, str]] = set()
    for ticker in tickers:
        pending = store.get_pending_reflections(horizon_days=0, as_of_date="2099-01-01")
        for row in pending:
            existing.add((row["ticker"], row["trade_date"]))
        # Also check reflected ones
        with store._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, trade_date FROM analysis_results"
            ).fetchall()
            for row in rows:
                existing.add((row["ticker"], row["trade_date"]))

    completed = 0
    skipped = 0

    for ticker in tickers:
        for date in dates:
            completed += 1

            if (ticker, date) in existing:
                skipped += 1
                logger.info(
                    "[%d/%d] SKIP %s %s (already done)", completed, total, ticker, date
                )
                continue

            logger.info(
                "[%d/%d] Analyzing %s on %s ...", completed, total, ticker, date
            )
            t0 = time.time()

            try:
                final_state, signal = ta.propagate(ticker, date)
                elapsed = time.time() - t0
                result = {
                    "ticker": ticker,
                    "date": date,
                    "signal": signal,
                    "elapsed": round(elapsed, 1),
                    "status": "ok",
                }
                results.append(result)
                logger.info(
                    "[%d/%d] %s %s → %s (%.1fs)",
                    completed, total, ticker, date, signal, elapsed,
                )

                # Save timing
                store.save_timing(ticker, date, elapsed)

            except Exception as e:
                elapsed = time.time() - t0
                result = {
                    "ticker": ticker,
                    "date": date,
                    "signal": None,
                    "elapsed": round(elapsed, 1),
                    "status": f"error: {e}",
                }
                results.append(result)
                logger.error(
                    "[%d/%d] %s %s FAILED: %s (%.1fs)",
                    completed, total, ticker, date, e, elapsed,
                )

    # Run T+N reflection on all eligible records
    logger.info("Running T+N reflection on eligible records...")
    reflect_results = ta.auto_reflect_pending()
    logger.info("Reflected %d records", len(reflect_results))

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] != "ok")
    logger.info("=" * 50)
    logger.info("Backtest complete: %d ok, %d failed, %d skipped", ok, fail, skipped)
    logger.info("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(description="TradingAgents backtest")
    parser.add_argument(
        "--tickers",
        default="NVDA",
        help="Comma-separated tickers (default: NVDA)",
    )
    parser.add_argument(
        "--start",
        default="2024-11-01",
        help="Start date yyyy-mm-dd (default: 2024-11-01)",
    )
    parser.add_argument(
        "--end",
        default="2025-03-14",
        help="End date yyyy-mm-dd (default: 2025-03-14)",
    )
    parser.add_argument(
        "--interval",
        choices=["daily", "weekly", "biweekly"],
        default="weekly",
        help="Analysis interval (default: weekly)",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    dates = generate_dates(args.start, args.end, args.interval)

    logger.info("Backtest: %s × %d dates (%s)", tickers, len(dates), args.interval)
    logger.info("Date range: %s → %s", dates[0] if dates else "N/A", dates[-1] if dates else "N/A")

    estimated_cost = len(tickers) * len(dates) * 0.03
    logger.info("Estimated cost: ~$%.2f (gpt-4o-mini)", estimated_cost)

    config = build_backtest_config()
    results = run_backtest(tickers, dates, config)

    # Save results to JSON
    output_path = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
