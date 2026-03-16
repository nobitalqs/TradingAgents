"""Tests for tradingagents.orchestrator.scheduler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.hooks.base import HookContext, HookEvent
from tradingagents.orchestrator.scheduler import ScheduledJob, TradingScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(jobs: list[dict] | None = None) -> dict:
    return {
        "scheduler": {
            "jobs": jobs
            or [
                {
                    "name": "morning_scan",
                    "cron": "30 9 * * 1-5",
                    "tickers": ["AAPL", "MSFT"],
                    "timezone": "US/Eastern",
                    "enabled": True,
                },
                {
                    "name": "disabled_job",
                    "cron": "0 12 * * *",
                    "tickers": ["GOOG"],
                    "enabled": False,
                },
            ]
        }
    }


@pytest.fixture()
def mock_ta() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_hooks() -> MagicMock:
    hook = MagicMock()
    hook.dispatch = MagicMock(side_effect=lambda ctx: ctx)
    return hook


@pytest.fixture()
def scheduler(mock_ta: MagicMock, mock_hooks: MagicMock) -> TradingScheduler:
    return TradingScheduler(
        config=_make_config(),
        trading_graph=mock_ta,
        hook_manager=mock_hooks,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadJobs:
    """load_jobs should parse config and register only enabled jobs."""

    def test_load_jobs_from_config(
        self, scheduler: TradingScheduler, mock_ta: MagicMock
    ) -> None:
        with patch.object(scheduler._scheduler, "add_job") as add_mock:
            scheduler.load_jobs()

        # Only enabled jobs are registered
        assert "morning_scan" in scheduler._jobs
        assert "disabled_job" not in scheduler._jobs
        assert add_mock.call_count == 1

        job = scheduler._jobs["morning_scan"]
        assert job.tickers == ["AAPL", "MSFT"]
        assert job.timezone == "US/Eastern"


class TestAddJob:
    """add_job should register a new job dynamically."""

    def test_add_job(self, scheduler: TradingScheduler) -> None:
        new_job = ScheduledJob(
            name="evening_scan",
            cron="0 16 * * 1-5",
            tickers=["TSLA"],
        )

        with patch.object(scheduler._scheduler, "add_job"):
            scheduler.add_job(new_job)

        assert "evening_scan" in scheduler._jobs
        assert scheduler._jobs["evening_scan"].tickers == ["TSLA"]


class TestRemoveJob:
    """remove_job should remove from internal dict and APScheduler."""

    def test_remove_job(self, scheduler: TradingScheduler) -> None:
        job = ScheduledJob(name="temp_job", cron="0 10 * * *", tickers=["AMD"])

        with patch.object(scheduler._scheduler, "add_job"):
            scheduler.add_job(job)

        with patch.object(scheduler._scheduler, "remove_job"):
            scheduler.remove_job("temp_job")

        assert "temp_job" not in scheduler._jobs

    def test_remove_nonexistent_raises(self, scheduler: TradingScheduler) -> None:
        with pytest.raises(KeyError, match="No job named"):
            scheduler.remove_job("does_not_exist")


class TestStatusProperty:
    """status should report running state, jobs, and recent runs."""

    def test_status_property(self, scheduler: TradingScheduler) -> None:
        with patch.object(scheduler._scheduler, "add_job"):
            scheduler.add_job(
                ScheduledJob(name="scan", cron="0 9 * * *", tickers=["NVDA"])
            )

        status = scheduler.status
        assert isinstance(status["running"], bool)
        assert "scan" in status["jobs"]
        assert status["jobs"]["scan"]["tickers"] == ["NVDA"]
        assert isinstance(status["recent_runs"], list)


class TestStartCallsLoadJobs:
    """start() must call load_jobs and then start the scheduler."""

    def test_start_calls_load_jobs(
        self, scheduler: TradingScheduler
    ) -> None:
        with (
            patch.object(scheduler, "load_jobs") as load_mock,
            patch.object(scheduler._scheduler, "start") as start_mock,
        ):
            scheduler.start()

        load_mock.assert_called_once()
        start_mock.assert_called_once()
