"""TradingAgents Enhanced — full orchestration stack entry point.

Startup sequence:
  1. Load configuration
  2. Initialize Hook system
  3. Initialize TradingGraph (with hook_manager)
  4. Start Heartbeat (if enabled)
  5. Start Scheduler (if enabled)
  6. Start MessageGateway (if enabled)

Usage:
  python main_enhanced.py
"""

import asyncio
import copy
import logging
import signal
from datetime import datetime

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.hooks.hook_manager import HookManager
from tradingagents.logging_config import setup_logging

load_dotenv()
setup_logging(verbosity=1)
logger = logging.getLogger("tradingagents.main")


def build_config() -> dict:
    """Build enhanced configuration. Override as needed."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    # LLM — adjust to your provider
    config["llm_provider"] = "openai"
    config["deep_think_llm"] = "gpt-5.2"
    config["quick_think_llm"] = "gpt-5-mini"

    # Hooks
    config["hooks"]["entries"]["journal"]["enabled"] = True
    config["hooks"]["entries"]["ratelimit"]["enabled"] = True
    config["hooks"]["entries"]["data_integrity"]["enabled"] = True

    # Scheduler (disabled by default — uncomment to enable)
    # config["scheduler"]["enabled"] = True
    # config["scheduler"]["jobs"] = [
    #     {
    #         "name": "pre_market_scan",
    #         "cron": "30 8 * * 1-5",
    #         "tickers": ["NVDA", "AAPL", "TSLA"],
    #     },
    # ]

    # Heartbeat (disabled by default)
    # config["heartbeat"]["enabled"] = True
    # config["heartbeat"]["watchlist"] = ["NVDA", "AAPL", "TSLA"]

    return config


async def main():
    config = build_config()

    # 1. Hook system
    hook_manager = HookManager(config)
    hook_manager.load_builtin_hooks()
    logger.info(f"Hooks loaded: {hook_manager.summary}")

    # 2. TradingGraph
    ta = TradingAgentsGraph(
        debug=False,
        config=config,
        hook_manager=hook_manager,
    )

    # 3. Heartbeat (optional)
    heartbeat = None
    if config.get("heartbeat", {}).get("enabled"):
        try:
            from tradingagents.orchestrator.heartbeat import MarketHeartbeat

            def on_alert(ticker, alert_type, details):
                logger.info(f"Heartbeat alert: {ticker} ({alert_type})")
                today = datetime.now().strftime("%Y-%m-%d")
                asyncio.create_task(
                    asyncio.to_thread(
                        ta.propagate, ticker, today,
                        {"alert_type": alert_type, "alert_details": details},
                    )
                )

            heartbeat = MarketHeartbeat(config, hook_manager, on_alert)
        except ImportError:
            logger.warning("Heartbeat module not available (install apscheduler)")

    # 4. Scheduler (optional)
    scheduler = None
    if config.get("scheduler", {}).get("enabled"):
        try:
            from tradingagents.orchestrator.scheduler import TradingScheduler
            scheduler = TradingScheduler(config, ta, hook_manager)
        except ImportError:
            logger.warning("Scheduler module not available (install apscheduler)")

    # 5. Message Gateway (optional)
    gateway_runner = None
    if config.get("message_gateway", {}).get("enabled"):
        try:
            from aiohttp import web
            from tradingagents.orchestrator.message_gateway import MessageGateway
            gateway = MessageGateway(config, ta, hook_manager, heartbeat)
            gateway_runner = web.AppRunner(gateway._app)
            await gateway_runner.setup()
            host = config["message_gateway"]["host"]
            port = config["message_gateway"]["port"]
            site = web.TCPSite(gateway_runner, host, port)
            await site.start()
            logger.info(f"Gateway started on {host}:{port}")
        except ImportError:
            logger.warning("Gateway module not available (install aiohttp)")

    # Start services
    tasks = []
    if heartbeat:
        tasks.append(asyncio.create_task(heartbeat.start()))
    if scheduler:
        scheduler.start()

    logger.info("=" * 50)
    logger.info("TradingAgents Enhanced is running")
    logger.info(f"  Hooks: {hook_manager.summary['total']} loaded")
    logger.info(f"  Heartbeat: {'ON' if heartbeat else 'OFF'}")
    logger.info(f"  Scheduler: {'ON' if scheduler else 'OFF'}")
    logger.info(f"  Gateway: {'ON' if gateway_runner else 'OFF'}")
    logger.info("=" * 50)

    # Wait for shutdown signal
    stop = asyncio.Event()

    def handle_signal():
        logger.info("Shutting down...")
        stop.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await stop.wait()

    # Cleanup
    if heartbeat:
        heartbeat.stop()
    if scheduler:
        scheduler.stop()
    if gateway_runner:
        await gateway_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
