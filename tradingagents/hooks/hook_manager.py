"""HookManager: synchronous hook dispatch with error isolation."""

from __future__ import annotations

import logging
from collections import defaultdict

from tradingagents.hooks.base import BaseHook, HookContext, HookEvent

logger = logging.getLogger(__name__)


class HookManager:
    """Registry and synchronous dispatcher for lifecycle hooks.

    Hooks are dispatched in registration order.  A failing hook logs an
    error and the chain continues.  If any hook sets ``context.skip = True``
    the remaining hooks for that event are skipped.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config if config is not None else {}
        self._hooks: defaultdict[HookEvent, list[BaseHook]] = defaultdict(list)
        self._all_hooks: list[BaseHook] = []

    # ── registration ─────────────────────────────────────────────

    def register(self, hook: BaseHook) -> None:
        """Register a hook for all of its subscribed events."""
        self._all_hooks.append(hook)
        for event in hook.subscriptions:
            self._hooks[event].append(hook)

    def unregister(self, hook_name: str) -> None:
        """Remove all hooks whose name matches *hook_name*."""
        self._all_hooks = [h for h in self._all_hooks if h.name != hook_name]
        for event in self._hooks:
            self._hooks[event] = [
                h for h in self._hooks[event] if h.name != hook_name
            ]

    # ── dispatch ─────────────────────────────────────────────────

    def dispatch(self, context: HookContext) -> HookContext:
        """Run every hook subscribed to ``context.event`` in order.

        Returns the (possibly transformed) context.  Exceptions inside a
        hook are logged but do not halt the chain.
        """
        ctx = context
        for hook in self._hooks[ctx.event]:
            try:
                ctx = hook.handle(ctx)
            except Exception:
                logger.error(
                    "Hook %r failed for event %s",
                    hook.name,
                    ctx.event,
                    exc_info=True,
                )
                continue

            if ctx.skip:
                break

        return ctx

    # ── builtin loading ──────────────────────────────────────────

    def load_builtin_hooks(self) -> None:
        """Load and register builtin hooks based on config.

        Expected config shape::

            {
                "hooks": {
                    "entries": {
                        "journal": {"enabled": True, ...},
                        "ratelimit": {"enabled": True, ...},
                        ...
                    }
                }
            }
        """
        entries: dict = (
            self._config.get("hooks", {}).get("entries", {})
        )

        _builtin_map: dict[str, tuple[str, str]] = {
            "journal": (
                "tradingagents.hooks.builtin.journal_hook",
                "JournalHook",
            ),
            "ratelimit": (
                "tradingagents.hooks.builtin.ratelimit_hook",
                "RateLimitHook",
            ),
            "portfolio_context": (
                "tradingagents.hooks.builtin.portfolio_hook",
                "PortfolioContextHook",
            ),
            "data_integrity": (
                "tradingagents.hooks.builtin.integrity_hook",
                "DataIntegrityHook",
            ),
            "notify": (
                "tradingagents.hooks.builtin.notify_hook",
                "NotifyHook",
            ),
            "auto_reflect": (
                "tradingagents.hooks.builtin.memory_hook",
                "AutoReflectHook",
            ),
        }

        for name, entry_cfg in entries.items():
            if not entry_cfg.get("enabled", False):
                continue
            if name not in _builtin_map:
                logger.warning("Unknown builtin hook: %s", name)
                continue

            module_path, class_name = _builtin_map[name]
            try:
                import importlib

                mod = importlib.import_module(module_path)
                hook_cls = getattr(mod, class_name)
                self.register(hook_cls(config=entry_cfg))
            except Exception:
                logger.error(
                    "Failed to load builtin hook %s",
                    name,
                    exc_info=True,
                )

    # ── introspection ────────────────────────────────────────────

    @property
    def summary(self) -> dict:
        """Return a summary of registered hooks."""
        return {
            "total": len(self._all_hooks),
            "hooks": [
                {
                    "name": h.name,
                    "enabled": True,
                    "events": [e.value for e in h.subscriptions],
                }
                for h in self._all_hooks
            ],
        }
