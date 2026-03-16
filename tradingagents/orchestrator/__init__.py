"""Orchestrator layer for TradingAgents.

Provides scheduling, market monitoring, and external gateway capabilities
on top of the core trading graph. Key components:

- TradingScheduler: APScheduler-based cron job runner for automated analysis.
- MarketHeartbeat: Real-time market anomaly detection (price/volume spikes).
- MessageGateway: Lightweight aiohttp REST API for ad-hoc analysis and control.
"""
