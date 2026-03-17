"""Integration tests: TradingScheduler + HookManager lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from tradingagents.hooks.base import HookContext, HookEvent
from tradingagents.hooks.hook_manager import HookManager
from tradingagents.orchestrator.scheduler import ScheduledJob, TradingScheduler


@pytest.fixture
def hook_manager():
    return HookManager()


@pytest.fixture
def mock_graph():
    """Mock TradingAgentsGraph with a configurable propagate method."""
    graph = MagicMock()
    graph.propagate.return_value = ({"final_trade_decision": "BUY 100 NVDA"}, "BUY")
    return graph


@pytest.fixture
def scheduler(mock_graph, hook_manager):
    config = {"scheduler": {"jobs": []}}
    return TradingScheduler(config, mock_graph, hook_manager)


@pytest.fixture
def sample_job():
    return ScheduledJob(
        name="daily_tech",
        cron="0 9 * * 1-5",
        tickers=["NVDA", "AAPL"],
        max_retries=1,
        retry_backoff_seconds=[0],
    )


class TestSchedulerJobManagement:
    """Job add/remove/status lifecycle."""

    def test_add_job_appears_in_status(self, scheduler, sample_job):
        scheduler.add_job(sample_job)
        status = scheduler.status
        assert "daily_tech" in status["jobs"]
        assert status["jobs"]["daily_tech"]["tickers"] == ["NVDA", "AAPL"]

    def test_remove_job(self, scheduler, sample_job):
        scheduler.add_job(sample_job)
        scheduler.remove_job("daily_tech")
        assert "daily_tech" not in scheduler.status["jobs"]

    def test_remove_nonexistent_raises(self, scheduler):
        with pytest.raises(KeyError, match="ghost"):
            scheduler.remove_job("ghost")

    def test_load_jobs_from_config(self, mock_graph, hook_manager):
        config = {
            "scheduler": {
                "jobs": [
                    {
                        "name": "morning",
                        "cron": "30 9 * * 1-5",
                        "tickers": ["GOOG"],
                        "enabled": True,
                    },
                    {
                        "name": "disabled_job",
                        "cron": "0 10 * * *",
                        "tickers": ["MSFT"],
                        "enabled": False,
                    },
                ]
            }
        }
        sched = TradingScheduler(config, mock_graph, hook_manager)
        sched.load_jobs()
        assert "morning" in sched.status["jobs"]
        assert "disabled_job" not in sched.status["jobs"]


class TestSchedulerExecution:
    """_execute_job dispatches hooks and calls propagate."""

    @pytest.mark.asyncio
    async def test_execute_job_dispatches_hooks(
        self, scheduler, sample_job, hook_manager
    ):
        """CRON_JOB_START and CRON_JOB_END events are dispatched."""
        dispatched_events: list[HookEvent] = []
        original_dispatch = hook_manager.dispatch

        def tracking_dispatch(ctx: HookContext) -> HookContext:
            dispatched_events.append(ctx.event)
            return original_dispatch(ctx)

        hook_manager.dispatch = tracking_dispatch

        scheduler.add_job(sample_job)
        await scheduler._execute_job(sample_job)

        assert HookEvent.CRON_JOB_START in dispatched_events
        assert HookEvent.CRON_JOB_END in dispatched_events

    @pytest.mark.asyncio
    async def test_execute_job_calls_propagate_per_ticker(
        self, scheduler, sample_job, mock_graph
    ):
        scheduler.add_job(sample_job)
        await scheduler._execute_job(sample_job)

        assert mock_graph.propagate.call_count == 2
        tickers_called = [
            call.args[0] for call in mock_graph.propagate.call_args_list
        ]
        assert "NVDA" in tickers_called
        assert "AAPL" in tickers_called

    @pytest.mark.asyncio
    async def test_execute_job_records_recent_run(self, scheduler, sample_job):
        await scheduler._execute_job(sample_job)

        assert len(scheduler.status["recent_runs"]) == 1
        run = scheduler.status["recent_runs"][0]
        assert run["job_name"] == "daily_tech"
        assert "NVDA" in run["results"]
        assert "AAPL" in run["results"]

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, scheduler, mock_graph):
        """Failed propagate retries up to max_retries."""
        job = ScheduledJob(
            name="retry_test",
            cron="0 9 * * *",
            tickers=["FAIL"],
            max_retries=2,
            retry_backoff_seconds=[0, 0],
        )
        mock_graph.propagate.side_effect = RuntimeError("API down")

        scheduler.add_job(job)
        await scheduler._execute_job(job)

        # 1 initial + 2 retries = 3 total calls
        assert mock_graph.propagate.call_count == 3
        run = scheduler.status["recent_runs"][-1]
        assert run["results"]["FAIL"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_partial_failure(self, scheduler, mock_graph):
        """One ticker fails, others succeed — both recorded."""
        job = ScheduledJob(
            name="partial",
            cron="0 9 * * *",
            tickers=["NVDA", "FAIL"],
            max_retries=0,
            retry_backoff_seconds=[0],
        )

        def propagate_side_effect(ticker, date, **kwargs):
            if ticker == "FAIL":
                raise RuntimeError("API error")
            return ({"final_trade_decision": "BUY"}, "BUY")

        mock_graph.propagate.side_effect = propagate_side_effect

        scheduler.add_job(job)
        await scheduler._execute_job(job)

        run = scheduler.status["recent_runs"][-1]
        assert run["results"]["NVDA"]["status"] == "ok"
        assert run["results"]["FAIL"]["status"] == "error"
