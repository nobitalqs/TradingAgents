"""Lightweight aiohttp REST gateway for ad-hoc analysis and control."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date
from typing import Any

from aiohttp import web

from tradingagents.hooks.base import HookContext, HookEvent

logger = logging.getLogger(__name__)


class MessageGateway:
    """HTTP API for triggering analysis and inspecting system state.

    Routes
    ------
    POST /analyze          Submit a ticker for background analysis.
    POST /watchlist/add    Add a ticker to the heartbeat watchlist.
    POST /watchlist/remove Remove a ticker from the heartbeat watchlist.
    GET  /status           Scheduler + watchlist summary.
    GET  /health           Simple liveness probe.

    Parameters
    ----------
    config : dict
        Full application config; gateway reads ``config["message_gateway"]``.
    trading_graph : TradingAgentsGraph
        The graph whose ``propagate`` is invoked by ``/analyze``.
    hook_manager : object
        Synchronous ``dispatch(ctx) -> HookContext``.
    heartbeat : MarketHeartbeat | None
        Optional heartbeat reference for watchlist management.
    """

    def __init__(
        self,
        config: dict[str, Any],
        trading_graph: Any,
        hook_manager: Any,
        heartbeat: Any | None = None,
    ) -> None:
        gw_cfg: dict[str, Any] = config.get("message_gateway", {})

        self._ta = trading_graph
        self._hook_manager = hook_manager
        self._heartbeat = heartbeat

        self._host: str = gw_cfg.get("host", "0.0.0.0")
        self._port: int = gw_cfg.get("port", 8080)
        self._auth_token: str = gw_cfg.get(
            "auth_token", os.environ.get("TRADINGAGENTS_GATEWAY_TOKEN", "")
        )

        self._app = web.Application(middlewares=[self._auth_middleware])
        self._app.router.add_post("/analyze", self._handle_analyze)
        self._app.router.add_post("/watchlist/add", self._handle_watchlist_add)
        self._app.router.add_post("/watchlist/remove", self._handle_watchlist_remove)
        self._app.router.add_get("/status", self._handle_status)
        self._app.router.add_get("/health", self._handle_health)

        self._runner: web.AppRunner | None = None
        self._background_tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the HTTP server."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info("MessageGateway listening on %s:%s", self._host, self._port)

    async def stop(self) -> None:
        """Shut down the HTTP server and cancel background tasks."""
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()

        if self._runner is not None:
            await self._runner.cleanup()

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    @web.middleware
    async def _auth_middleware(
        self,
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        """Bearer-token authentication middleware.

        Skips auth when ``_auth_token`` is empty (development mode).
        The ``/health`` endpoint is always unauthenticated.
        """
        if request.path == "/health" or not self._auth_token:
            return await handler(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return web.json_response(
                {"error": "Missing or malformed Authorization header"}, status=401
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token != self._auth_token:
            return web.json_response({"error": "Invalid token"}, status=403)

        return await handler(request)

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def _handle_analyze(self, request: web.Request) -> web.Response:
        """Accept a ticker for background analysis (202 Accepted)."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        ticker = body.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return web.json_response(
                {"error": "'ticker' is required and must be a string"}, status=400
            )

        analysis_date = body.get("date", str(date.today()))

        self._hook_manager.dispatch(
            HookContext(
                event=HookEvent.BEFORE_PROPAGATE,
                metadata={"ticker": ticker, "date": analysis_date, "source": "gateway"},
            )
        )

        task = asyncio.create_task(self._run_analysis(ticker, analysis_date))
        self._background_tasks.append(task)
        task.add_done_callback(lambda t: self._background_tasks.remove(t))

        return web.json_response(
            {"status": "accepted", "ticker": ticker, "date": analysis_date},
            status=202,
        )

    async def _handle_watchlist_add(self, request: web.Request) -> web.Response:
        """Add a ticker to the heartbeat watchlist."""
        if self._heartbeat is None:
            return web.json_response(
                {"error": "Heartbeat not configured"}, status=503
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        ticker = body.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return web.json_response(
                {"error": "'ticker' is required and must be a string"}, status=400
            )

        watchlist: list[str] = self._heartbeat._watchlist
        if ticker not in watchlist:
            watchlist.append(ticker)

        return web.json_response(
            {"status": "ok", "watchlist": list(watchlist)}
        )

    async def _handle_watchlist_remove(self, request: web.Request) -> web.Response:
        """Remove a ticker from the heartbeat watchlist."""
        if self._heartbeat is None:
            return web.json_response(
                {"error": "Heartbeat not configured"}, status=503
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        ticker = body.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return web.json_response(
                {"error": "'ticker' is required and must be a string"}, status=400
            )

        watchlist: list[str] = self._heartbeat._watchlist
        try:
            watchlist.remove(ticker)
        except ValueError:
            pass

        return web.json_response(
            {"status": "ok", "watchlist": list(watchlist)}
        )

    async def _handle_status(self, _request: web.Request) -> web.Response:
        """Return combined status of scheduler, heartbeat, and hooks."""
        payload: dict[str, Any] = {
            "watchlist": (
                list(self._heartbeat._watchlist) if self._heartbeat else []
            ),
            "pending_alerts": (
                self._heartbeat.pending_events if self._heartbeat else []
            ),
            "background_tasks": len(self._background_tasks),
        }
        return web.json_response(payload)

    async def _handle_health(self, _request: web.Request) -> web.Response:
        """Simple liveness check."""
        return web.json_response({"status": "ok"})

    # ------------------------------------------------------------------
    # Background work
    # ------------------------------------------------------------------

    async def _run_analysis(self, ticker: str, analysis_date: str) -> None:
        """Run ``propagate`` in a thread to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, self._ta.propagate, ticker, analysis_date
            )
            logger.info("Analysis complete for %s on %s", ticker, analysis_date)
        except Exception:
            logger.exception("Background analysis failed for %s", ticker)
            result = None

        self._hook_manager.dispatch(
            HookContext(
                event=HookEvent.AFTER_PROPAGATE,
                metadata={
                    "ticker": ticker,
                    "date": analysis_date,
                    "success": result is not None,
                },
            )
        )
