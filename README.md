<div align="center">

# TradingAgents Enhanced

**多智能体股票分析系统 — 4 位分析师辩论，裁判拍板，系统从结果中学习**

基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 重写，新增反思学习、生产部署与评估框架

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-319%20passed-brightgreen.svg)](#测试)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 功能概览

```
每个工作日 21:30（北京时间）自动执行：
  4 位分析师 → 多空辩论 → 交易员 → 风险辩论 → 最终决策
                                                    ↓
                                           飞书 / Slack 推送
                                                    ↓
                                         7 天后自动验证股价
                                         → 反思 → 更新记忆
```

你会在飞书收到一张彩色卡片（BUY=绿 / SELL=红 / HOLD=蓝），包含决策理由。系统会记住每次对错，并在未来分析中参考历史经验。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                  TradingAgents Enhanced                   │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ 技术分析  │  │ 情绪分析  │  │ 新闻分析  │  │ 基本面  │ │
│  │ Analyst  │  │ Analyst  │  │ Analyst  │  │ Analyst │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       └──────────────┼──────────────┼─────────────┘      │
│                      ▼                                   │
│               信号共识投票                                 │
│                      ▼                                   │
│          Bull ◄──► Bear（多空辩论）                       │
│                      ▼                                   │
│              Research Manager                            │
│                      ▼                                   │
│                   Trader                                 │
│                      ▼                                   │
│     激进派 ◄──► 保守派 ◄──► 中性派（风险辩论）             │
│                      ▼                                   │
│              Risk Judge → 最终决策                        │
└──────────┬───────────┬───────────┬───────────────────────┘
           │           │           │
     ┌─────▼──┐  ┌─────▼──┐  ┌────▼────┐
     │  飞书   │  │ SQLite │  │  日志   │
     │  推送   │  │  记忆  │  │  归档   │
     └────────┘  └────────┘  └─────────┘
```

## 与原版 TradingAgents 的差异

| 能力 | 原版 | 本项目 |
|------|------|--------|
| 反思学习 | 无 | T+7 自动验证股价 + LLM 反思 + 记忆更新 |
| 记忆持久化 | 纯内存，重启丢失 | SQLite + BM25 双向同步，FIFO 淘汰 |
| 消息推送 | 无 | 飞书彩色卡片 / Slack / Webhook |
| 定时调度 | 无 | APScheduler cron 定时任务 |
| HTTP API | 无 | REST 接口，随时触发分析 |
| 市场监控 | 无 | 心跳检测（价格/成交量异常告警） |
| 数据验证 | 无 | 多源价格校验 + 新闻可信度评分 |
| Hook 系统 | 无 | 16 种生命周期事件，6 个内置 Hook |
| 信号提取 | 简单正则 | 4 级降级（正则 → LLM → 严格模式 → 报错） |
| 分析师共识 | 无 | 方向投票 + 置信度评分 |
| 节点重试 | 无 | RetryPolicy 指数退避 + 抖动 |
| 评估框架 | 无 | 方向准确率、置信度校准、滚动趋势 |
| 测试覆盖 | 极少 | 319 个测试（单元 + 集成） |

## 快速开始

### 1. 安装

```bash
git clone https://github.com/nobitalqs/TradingAgents.git
cd TradingAgents
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 配置环境变量

```bash
# 必需
export OPENAI_API_KEY="sk-..."

# 飞书推送（可选）
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/..."
export FEISHU_WEBHOOK_SECRET="your-secret"

# 自定义（可选）
export TRADINGAGENTS_TICKERS="NVDA,AAPL,TSLA"    # 分析标的，默认 NVDA,AAPL,TSLA
export TRADINGAGENTS_CRON="30 21 * * 1-5"         # cron 表达式，默认工作日 21:30
export TRADINGAGENTS_TIMEZONE="Asia/Shanghai"      # 时区，默认北京时间
```

### 3. 启动

```bash
# 启动守护进程（定时调度 + HTTP 接口）
python main_enhanced.py

# 或手动触发单次分析
curl -X POST http://localhost:8899/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA"}'
```

### 4. 回测与评估

```bash
# 回测（仅使用 LLM 训练截止日期之后的数据）
python scripts/backtest.py \
  --tickers NVDA \
  --start 2024-11-01 \
  --end 2025-03-14 \
  --interval weekly

# 查看评估报告
python scripts/evaluate.py --ticker NVDA
```

## 评估示例

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

## 数据来源

默认全部免费：

| 数据源 | 提供数据 | 费用 |
|--------|---------|------|
| **yfinance** | 股价、技术指标、财报、新闻、内部人交易 | 免费 |
| **stockstats** | 技术指标计算（SMA/EMA/MACD/RSI/布林带/ATR） | 免费（本地计算） |
| **Alpha Vantage** | 同类数据（可选备用源） | 免费层：5 次/分钟 |

LLM 成本：gpt-4o-mini 约 $0.03/股/次。每日分析 3 只股票，月成本约 $1-2。

## 项目结构

```
tradingagents/
├── agents/           # 4 位分析师、多空研究员、交易员、风险辩论者
├── graph/            # LangGraph 图编排、信号处理、反思系统
├── hooks/            # 16 种生命周期事件、6 个内置 Hook
├── notify/           # 飞书、Slack、Webhook 通知器
├── orchestrator/     # 定时调度、心跳监控、HTTP 网关
├── verification/     # 数据可信度评分
├── learning/         # SQLite 持久化、自动反思、评估框架
└── dataflows/        # yfinance + Alpha Vantage 数据层

scripts/
├── backtest.py       # 历史回测
└── evaluate.py       # 评估指标计算
```

## 已知局限

- **BUY 偏见**：系统约 75% 的时间发出 BUY 信号，Bear Researcher 的 prompt 需要重新平衡
- **置信度标定反转**：HIGH 置信度信号准确率反而低于 LOW，共识映射规则需调整
- **无仓位管理**：仅输出 BUY/SELL/HOLD 方向，不包含仓位比例和止损位
- **LLM 数据泄露**：回测日期若在 LLM 训练数据范围内则结果不可靠，应仅使用训练截止日期之后的数据

## 致谢

本项目基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 开发。原始论文：[arXiv:2412.20138](https://arxiv.org/abs/2412.20138)。

## 免责声明

本项目仅供研究和个人使用，**不构成任何投资建议**。交易决策应由人类做出，而非自动化系统。使用风险自负。

## 许可证

与原始 TradingAgents 项目一致，详见 [LICENSE](LICENSE)。
