"""PortfolioContextHook: injects portfolio holdings into the propagation context."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent

logger = logging.getLogger(__name__)


class PortfolioContextHook(BaseHook):
    """Read a portfolio JSON file and inject a summary into ctx.inject_context.

    Config keys:
        portfolio_file (str): path to a JSON file with holdings.
    """

    name = "portfolio_context"
    subscriptions = [HookEvent.BEFORE_PROPAGATE]

    def handle(self, context: HookContext) -> HookContext:
        portfolio_file = self.config.get("portfolio_file")
        if not portfolio_file:
            return context

        path = Path(portfolio_file)
        if not path.exists():
            logger.warning("Portfolio file not found: %s", portfolio_file)
            return context

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.error("Failed to read portfolio file: %s", portfolio_file, exc_info=True)
            return context

        holdings = data.get("holdings", [])
        if not holdings:
            return context

        lines = ["Current portfolio holdings:"]
        for h in holdings:
            symbol = h.get("symbol", "???")
            qty = h.get("quantity", 0)
            avg_cost = h.get("avg_cost", 0.0)
            lines.append(f"  {symbol}: {qty} shares @ ${avg_cost:.2f}")

        total_value = data.get("total_value")
        if total_value is not None:
            lines.append(f"  Total portfolio value: ${total_value:,.2f}")

        inject_text = "\n".join(lines)
        return replace(context, inject_context=inject_text)
