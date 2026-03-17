# AlphaAgents

Multi-agent stock trading advisor — 4 analysts debate, a judge decides, the system learns from outcomes.

Built on [TradingAgents](https://github.com/TauricResearch/TradingAgents) by TauricResearch, rewritten with production deployment, reflection learning, and evaluation framework.

## What It Does

```
Every trading day at 21:30 (Beijing time):
  4 Analysts → Bull/Bear Debate → Trader → Risk Debate → Final Decision
                                                              ↓
                                                     Feishu/Slack push
                                                              ↓
                                                  7 days later: verify
                                                  actual stock price →
                                                  reflect → update memory
```

You receive a colored card on Feishu (or Slack) with BUY / SELL / HOLD and the reasoning. The system remembers what it got right and wrong, and uses that memory in future analyses.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     AlphaAgents                          │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Market   │  │ Social   │  │ News     │  │ Funds   │ │
│  │ Analyst  │  │ Analyst  │  │ Analyst  │  │ Analyst │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       └──────────────┼──────────────┼─────────────┘      │
│                      ▼                                   │
│              Signal Consensus                            │
│                      ▼                                   │
│         Bull ◄──► Bear  (debate)                         │
│                      ▼                                   │
│             Research Manager                             │
│                      ▼                                   │
│                   Trader                                 │
│                      ▼                                   │
│    Aggressive ◄──► Conservative ◄──► Neutral (debate)    │
│                      ▼                                   │
│                 Risk Judge → FINAL DECISION               │
└──────────┬───────────┬───────────┬───────────────────────┘
           │           │           │
     ┌─────▼──┐  ┌─────▼──┐  ┌────▼────┐
     │ Feishu │  │ SQLite │  │ Journal │
     │  Push  │  │ Memory │  │  Logs   │
     └────────┘  └────────┘  └─────────┘
```

## Differences from Original TradingAgents

| Feature | TauricResearch/TradingAgents | AlphaAgents |
|---------|------------------------------|-------------|
| Reflection learning | No | T+7 auto-reflection with yfinance price verification |
| Memory persistence | In-memory only, lost on restart | SQLite + BM25 with FIFO eviction |
| Notifications | No | Feishu cards / Slack / Webhook |
| Scheduling | No | APScheduler cron jobs |
| HTTP API | No | REST gateway for on-demand analysis |
| Market monitoring | No | Heartbeat (price/volume anomaly detection) |
| Data verification | No | Multi-source price validation + news credibility scoring |
| Hook system | No | 16 lifecycle events, 6 builtin hooks |
| Signal extraction | Simple regex | 4-level fallback (regex → LLM → strict → error) |
| Analyst consensus | No | Direction voting + confidence scoring |
| Node retry | No | RetryPolicy with exponential backoff + jitter |
| Evaluation | No | Direction accuracy, confidence calibration, rolling trend |
| Tests | Minimal | 319 tests (unit + integration) |

## Quick Start

### 1. Install

```bash
git clone https://github.com/nobitalqs/AlphaAgents.git
cd AlphaAgents
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Feishu notifications (optional)
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/..."
export FEISHU_WEBHOOK_SECRET="your-secret"

# Customization (optional)
export TRADINGAGENTS_TICKERS="NVDA,AAPL,TSLA"        # default: NVDA,AAPL,TSLA
export TRADINGAGENTS_CRON="30 21 * * 1-5"             # default: 21:30 Beijing time
export TRADINGAGENTS_TIMEZONE="Asia/Shanghai"          # default: Asia/Shanghai
```

### 3. Run

```bash
# Start the daemon (scheduler + gateway)
python main_enhanced.py

# Or trigger a single analysis
curl -X POST http://localhost:8899/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA"}'
```

### 4. Backtest

```bash
# Run on historical dates (post GPT-4o training cutoff)
python scripts/backtest.py --tickers NVDA --start 2024-11-01 --end 2025-03-14 --interval weekly

# Evaluate results
python scripts/evaluate.py --ticker NVDA
```

## Evaluation Example

```
==================================================
TradingAgents Evaluation Report
==================================================

Total records:          19
Directional (BUY+SELL): 16
HOLD:                   3

── Direction Accuracy ──
Overall:  62.5%
BUY:      57.1% (14 signals)
SELL:     100.0% (2 signals)

── Average Return by Signal ──
BUY   :  +0.05%
HOLD  :  -0.49%
SELL  :  -2.89%

── Confidence Calibration ──
HIGH    : 0.0% accuracy (1 decisions)
LOW     : 83.0% accuracy (6 decisions)
MEDIUM  : 56.0% accuracy (9 decisions)

── Rolling Accuracy ──
2025-01-17  ████████████████░░░░  80%
2025-01-24  ████████████░░░░░░░░  60%
2025-01-31  ████████████░░░░░░░░  60%
2025-02-07  ████████████████░░░░  80%
2025-02-21  ████████████░░░░░░░░  60%
2025-03-14  ████████░░░░░░░░░░░░  40%
==================================================
```

## Data Sources

All data is free by default:

| Source | Data | Cost |
|--------|------|------|
| **yfinance** | Stock prices, technicals, fundamentals, news, insider transactions | Free |
| **stockstats** | Technical indicator calculations (SMA, EMA, MACD, RSI, Bollinger, ATR) | Free (local) |
| **Alpha Vantage** | Same categories (optional fallback) | Free tier: 5 req/min |

LLM cost with gpt-4o-mini: ~$0.03/stock/analysis. Daily analysis of 3 stocks costs ~$1-2/month.

## Project Structure

```
tradingagents/
├── agents/           # 4 analysts, bull/bear researchers, trader, risk debaters
├── graph/            # LangGraph orchestration, signal processing, reflection
├── hooks/            # 16 lifecycle events, 6 builtin hooks
├── notify/           # Feishu, Slack, Webhook notifiers
├── orchestrator/     # Scheduler, heartbeat monitor, HTTP gateway
├── verification/     # Data credibility scoring
├── learning/         # SQLite persistence, auto-reflection, evaluation
└── dataflows/        # yfinance + Alpha Vantage data layer

scripts/
├── backtest.py       # Historical date-range analysis
└── evaluate.py       # Compute accuracy metrics from results
```

## Known Limitations

- **BUY bias**: System issues BUY signals ~75% of the time. Bear researcher prompts may need rebalancing.
- **Confidence calibration inverted**: HIGH confidence signals are less accurate than LOW.
- **No position sizing**: Outputs BUY/SELL/HOLD direction only, no allocation percentages or stop-loss levels.
- **LLM data leakage**: Backtesting on dates within LLM training data is unreliable. Only use dates after the model's training cutoff.

## Acknowledgments

Based on [TradingAgents](https://github.com/TauricResearch/TradingAgents) by [TauricResearch](https://github.com/TauricResearch). Original paper: [arXiv:2412.20138](https://arxiv.org/abs/2412.20138).

## Disclaimer

This is a research project and personal tool. It is **not financial advice**. Trading decisions should be made by humans, not automated systems. Use at your own risk.

## License

Same as the original TradingAgents project — see [LICENSE](LICENSE).
