"""RateLimitHook: sliding-window rate limiter for tool calls."""

from __future__ import annotations

import logging
import time

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent

logger = logging.getLogger(__name__)


class RateLimitHook(BaseHook):
    """Enforce a per-second rate limit on tool calls.

    Config keys:
        max_calls_per_second (int): allowed calls per window (default 10).
        window_seconds (float): sliding window size (default 1.0).
    """

    name = "ratelimit"
    subscriptions = [HookEvent.BEFORE_TOOL_CALL, HookEvent.AFTER_TOOL_CALL]

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        self._max_calls: int = self.config.get("max_calls_per_second", 10)
        self._window: float = self.config.get("window_seconds", 1.0)
        self._call_times: list[float] = []

        # Cumulative stats
        self._total_calls: int = 0
        self._total_errors: int = 0
        self._total_ms: float = 0.0

    def handle(self, context: HookContext) -> HookContext:
        if context.event == HookEvent.BEFORE_TOOL_CALL:
            return self._handle_before(context)
        return self._handle_after(context)

    # ── before: enforce rate limit ───────────────────────────────

    def _handle_before(self, context: HookContext) -> HookContext:
        now = time.monotonic()

        # Prune calls outside the window
        cutoff = now - self._window
        self._call_times = [t for t in self._call_times if t > cutoff]

        if len(self._call_times) >= self._max_calls:
            oldest = self._call_times[0]
            sleep_for = self._window - (now - oldest)
            if sleep_for > 0:
                logger.warning(
                    "Rate limit reached (%d/%d in %.1fs). Sleeping %.3fs.",
                    len(self._call_times),
                    self._max_calls,
                    self._window,
                    sleep_for,
                )
                time.sleep(sleep_for)

        self._call_times.append(time.monotonic())
        return context

    # ── after: record stats ──────────────────────────────────────

    def _handle_after(self, context: HookContext) -> HookContext:
        self._total_calls += 1
        duration = context.metadata.get("duration_ms", 0.0)
        self._total_ms += duration

        if context.metadata.get("error"):
            self._total_errors += 1

        return context

    # ── introspection ────────────────────────────────────────────

    @property
    def stats_summary(self) -> dict:
        """Return cumulative tool-call statistics."""
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_ms": self._total_ms,
        }
