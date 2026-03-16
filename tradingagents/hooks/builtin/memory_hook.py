"""AutoReflectHook: triggers reflection after a decision when returns data is available."""

from __future__ import annotations

import logging

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent

logger = logging.getLogger(__name__)


class AutoReflectHook(BaseHook):
    """Automatically invoke ``graph.reflect_and_remember()`` after a decision.

    Requires metadata to contain:
        recent_returns (list): recent PnL data to reflect on.
        trading_graph_ref: a reference to the TradingGraph instance.
    """

    name = "auto_reflect"
    subscriptions = [HookEvent.AFTER_DECISION]

    def handle(self, context: HookContext) -> HookContext:
        recent_returns = context.metadata.get("recent_returns")
        graph_ref = context.metadata.get("trading_graph_ref")

        if recent_returns is None or graph_ref is None:
            logger.debug(
                "AutoReflectHook skipped: missing recent_returns or trading_graph_ref"
            )
            return context

        try:
            graph_ref.reflect_and_remember(recent_returns)
            logger.info(
                "Auto-reflection completed for %s on %s",
                context.ticker,
                context.trade_date,
            )
        except Exception:
            logger.error(
                "Auto-reflection failed for %s",
                context.ticker,
                exc_info=True,
            )

        return context
