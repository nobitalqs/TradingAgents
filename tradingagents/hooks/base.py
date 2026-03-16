"""Base types for the hook system: HookEvent enum, HookContext dataclass, BaseHook ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class HookEvent(str, Enum):
    """Lifecycle events that hooks can subscribe to."""

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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class HookContext:
    """Immutable-style context passed through the hook chain.

    Hooks should return a new HookContext (via dataclasses.replace) rather
    than mutating this instance in-place.
    """

    event: HookEvent
    timestamp: datetime = field(default_factory=_utcnow)
    ticker: str = ""
    trade_date: str = ""
    state: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    inject_context: str | None = None
    skip: bool = False


class BaseHook(ABC):
    """Abstract base class for all hooks.

    Subclasses must define:
        name: str           – unique identifier for this hook
        subscriptions: list[HookEvent] – which events to listen for
        handle(context) -> HookContext – the hook logic
    """

    name: str
    subscriptions: list[HookEvent]

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = config if config is not None else {}

    @abstractmethod
    def handle(self, context: HookContext) -> HookContext:
        """Process a hook event and return a (possibly new) HookContext."""
        ...

    def __repr__(self) -> str:
        events = ", ".join(e.value for e in self.subscriptions)
        return f"<{self.__class__.__name__} name={self.name!r} events=[{events}]>"
