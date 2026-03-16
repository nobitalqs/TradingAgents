"""APScheduler-based cron scheduler for automated trading analysis."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from tradingagents.hooks.base import HookContext, HookEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledJob:
    """Immutable description of a recurring analysis job."""

    name: str
    cron: str
    tickers: list[str]
    timezone: str = "US/Eastern"
    session_mode: str = "isolated"
    enabled: bool = True
    max_retries: int = 3
    retry_backoff_seconds: list[int] = field(default_factory=lambda: [30, 60, 300])
    notify: bool = True


class TradingScheduler:
    """Manages cron-based scheduled analysis jobs via APScheduler.

    Parameters
    ----------
    config : dict
        Full application config; scheduler reads ``config["scheduler"]["jobs"]``.
    trading_graph : TradingAgentsGraph
        The core graph instance whose ``propagate`` method will be called.
    hook_manager : object
        Object with a synchronous ``dispatch(ctx: HookContext) -> HookContext``
        method used to emit lifecycle events.
    """

    def __init__(
        self,
        config: dict[str, Any],
        trading_graph: Any,
        hook_manager: Any,
    ) -> None:
        self._config = config
        self._ta = trading_graph
        self._hook_manager = hook_manager
        self._scheduler = AsyncIOScheduler()
        self._jobs: dict[str, ScheduledJob] = {}
        self._recent_runs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Job loading
    # ------------------------------------------------------------------

    def load_jobs(self) -> None:
        """Parse scheduler config and register each enabled job."""
        scheduler_cfg = self._config.get("scheduler", {})
        raw_jobs: list[dict[str, Any]] = scheduler_cfg.get("jobs", [])

        for raw in raw_jobs:
            job = ScheduledJob(
                name=raw["name"],
                cron=raw["cron"],
                tickers=list(raw["tickers"]),
                timezone=raw.get("timezone", "US/Eastern"),
                session_mode=raw.get("session_mode", "isolated"),
                enabled=raw.get("enabled", True),
                max_retries=raw.get("max_retries", 3),
                retry_backoff_seconds=list(
                    raw.get("retry_backoff_seconds", [30, 60, 300])
                ),
                notify=raw.get("notify", True),
            )
            if job.enabled:
                self._register_job(job)

    # ------------------------------------------------------------------
    # Dynamic job management
    # ------------------------------------------------------------------

    def add_job(self, job: ScheduledJob) -> None:
        """Add and register a single job at runtime."""
        self._register_job(job)

    def remove_job(self, name: str) -> None:
        """Remove a job by name."""
        if name in self._jobs:
            try:
                self._scheduler.remove_job(name)
            except Exception:
                logger.debug("Job %s not found in scheduler during removal", name)
            del self._jobs[name]
        else:
            raise KeyError(f"No job named '{name}'")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load jobs from config and start the scheduler."""
        self.load_jobs()
        self._scheduler.start()

    def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        self._scheduler.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict[str, Any]:
        """Return scheduler health snapshot."""
        return {
            "running": self._scheduler.running,
            "jobs": {name: _job_to_dict(job) for name, job in self._jobs.items()},
            "recent_runs": list(self._recent_runs[-50:]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_job(self, job: ScheduledJob) -> None:
        """Create a CronTrigger and add the job to APScheduler."""
        self._jobs[job.name] = job

        parts = job.cron.split()
        trigger = CronTrigger(
            minute=parts[0] if len(parts) > 0 else "*",
            hour=parts[1] if len(parts) > 1 else "*",
            day=parts[2] if len(parts) > 2 else "*",
            month=parts[3] if len(parts) > 3 else "*",
            day_of_week=parts[4] if len(parts) > 4 else "*",
            timezone=job.timezone,
        )

        self._scheduler.add_job(
            self._execute_job,
            trigger=trigger,
            id=job.name,
            args=[job],
            replace_existing=True,
        )

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Run analysis for every ticker in *job* with retry + backoff."""
        today = str(date.today())

        self._hook_manager.dispatch(
            HookContext(
                event=HookEvent.CRON_JOB_START,
                metadata={"job_name": job.name, "tickers": job.tickers, "date": today},
            )
        )

        results: dict[str, Any] = {}

        for ticker in job.tickers:
            result = await self._run_ticker_with_retry(ticker, today, job)
            results[ticker] = result

        run_record = {
            "job_name": job.name,
            "date": today,
            "results": results,
            "timestamp": time.time(),
        }
        self._recent_runs.append(run_record)

        self._hook_manager.dispatch(
            HookContext(
                event=HookEvent.CRON_JOB_END,
                metadata={
                    "job_name": job.name,
                    "tickers": job.tickers,
                    "date": today,
                    "results": results,
                },
            )
        )

    async def _run_ticker_with_retry(
        self, ticker: str, today: str, job: ScheduledJob
    ) -> dict[str, Any]:
        """Call ``propagate`` with exponential-ish backoff on failure."""
        last_error: Exception | None = None

        for attempt in range(job.max_retries + 1):
            try:
                result = self._ta.propagate(ticker, today)
                return {"status": "ok", "attempt": attempt + 1, "result": result}
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Job %s ticker %s attempt %d failed: %s",
                    job.name,
                    ticker,
                    attempt + 1,
                    exc,
                )
                if attempt < job.max_retries:
                    backoff = job.retry_backoff_seconds[
                        min(attempt, len(job.retry_backoff_seconds) - 1)
                    ]
                    await asyncio.sleep(backoff)

        return {
            "status": "error",
            "attempt": job.max_retries + 1,
            "error": str(last_error),
        }


def _job_to_dict(job: ScheduledJob) -> dict[str, Any]:
    """Serialize a ScheduledJob to a plain dict."""
    return {
        "name": job.name,
        "cron": job.cron,
        "tickers": job.tickers,
        "timezone": job.timezone,
        "session_mode": job.session_mode,
        "enabled": job.enabled,
        "max_retries": job.max_retries,
        "retry_backoff_seconds": job.retry_backoff_seconds,
        "notify": job.notify,
    }
