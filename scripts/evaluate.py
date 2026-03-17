#!/usr/bin/env python3
"""Evaluate TradingAgents decision quality from stored results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --ticker NVDA
    python scripts/evaluate.py --db ./tradingagents_memory.db
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.learning.evaluation import evaluate, format_report
from tradingagents.learning.persistence import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="Evaluate TradingAgents results")
    parser.add_argument("--ticker", default="", help="Filter by ticker (default: all)")
    parser.add_argument("--db", default="./tradingagents_memory.db", help="DB path")
    parser.add_argument("--window", type=int, default=5, help="Rolling accuracy window")
    args = parser.parse_args()

    store = MemoryStore(db_path=args.db)
    report = evaluate(store, ticker=args.ticker, rolling_window=args.window)

    if report.total_records == 0:
        print("No reflected results found. Run backtest + reflect first.")
        return

    print(format_report(report))


if __name__ == "__main__":
    main()
