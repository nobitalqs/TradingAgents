"""JournalHook: writes structured JSONL entries for analyst, debate, and decision events."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent

logger = logging.getLogger(__name__)

# Fields extracted from metadata per event type
_EVENT_FIELDS: dict[HookEvent, list[str]] = {
    HookEvent.AFTER_ANALYST: ["analyst_type"],
    HookEvent.AFTER_DEBATE: ["debate_type"],
    HookEvent.AFTER_DECISION: ["decision", "confidence"],
}


class JournalHook(BaseHook):
    """Append one JSONL line per subscribed event to a journal file.

    Config keys:
        output_dir (str): directory to write journal files into.
    """

    name = "journal"
    subscriptions = [
        HookEvent.AFTER_ANALYST,
        HookEvent.AFTER_DEBATE,
        HookEvent.AFTER_DECISION,
    ]

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        output_dir = self.config.get("output_dir", "./journal")
        self._output_path = Path(output_dir)
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._file = self._output_path / "journal.jsonl"

    def handle(self, context: HookContext) -> HookContext:
        entry: dict = {
            "event": context.event.value,
            "timestamp": context.timestamp.isoformat(),
            "ticker": context.ticker,
            "trade_date": context.trade_date,
        }

        # Pull event-specific fields from metadata
        extra_fields = _EVENT_FIELDS.get(context.event, [])
        for field_name in extra_fields:
            if field_name in context.metadata:
                entry[field_name] = context.metadata[field_name]

        # Also include any other metadata keys not already captured
        for key, value in context.metadata.items():
            if key not in entry:
                entry[key] = value

        try:
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            logger.error("Failed to write journal entry", exc_info=True)

        return context
