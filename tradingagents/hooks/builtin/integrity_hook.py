"""DataIntegrityHook: flags low-credibility data after analyst runs."""

from __future__ import annotations

import logging
from dataclasses import replace

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent

logger = logging.getLogger(__name__)

# Thresholds
_UNRELIABLE_THRESHOLD = 2
_WARNING_THRESHOLD = 1


class DataIntegrityHook(BaseHook):
    """Check metadata["data_credibility"] and inject a warning if quality is low.

    Triggers when:
        - unreliable_count >= _UNRELIABLE_THRESHOLD, or
        - warnings list has >= _WARNING_THRESHOLD entries
    """

    name = "data_integrity"
    subscriptions = [HookEvent.AFTER_ANALYST]

    def handle(self, context: HookContext) -> HookContext:
        credibility = context.metadata.get("data_credibility")
        if credibility is None:
            return context

        unreliable_count: int = credibility.get("unreliable_count", 0)
        warnings: list[str] = credibility.get("warnings", [])

        issues: list[str] = []
        if unreliable_count >= _UNRELIABLE_THRESHOLD:
            issues.append(
                f"{unreliable_count} unreliable data sources detected"
            )
        if len(warnings) >= _WARNING_THRESHOLD:
            issues.append(
                f"Data warnings: {'; '.join(warnings)}"
            )

        if not issues:
            return context

        warning_text = (
            "[DATA INTEGRITY WARNING] "
            + " | ".join(issues)
            + ". Treat downstream analysis with caution."
        )
        logger.warning(warning_text)

        return replace(context, inject_context=warning_text)
