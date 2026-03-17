"""Evaluation metrics for TradingAgents decision quality.

Reads reflected analysis_results from MemoryStore and computes:
1. Direction accuracy (overall + by signal type + signal counts)
2. Average return by signal type
3. Confidence calibration (accuracy per confidence level)
4. Rolling accuracy (trend over time)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tradingagents.learning.persistence import MemoryStore

logger = logging.getLogger(__name__)

_DIRECTION_SIGNALS = frozenset({"BUY", "SELL"})


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results."""

    total_records: int
    directional_records: int
    hold_records: int

    direction_accuracy: float
    buy_accuracy: float
    sell_accuracy: float
    signal_counts: dict[str, int]

    avg_return_by_signal: dict[str, float]

    accuracy_by_confidence: dict[str, dict[str, Any]]

    rolling_accuracy: list[tuple[str, float]]

    records: list[dict] = field(default_factory=list)


def evaluate(
    store: MemoryStore,
    ticker: str = "",
    rolling_window: int = 5,
) -> EvaluationReport:
    """Run evaluation on reflected results."""
    rows = store.get_reflected_results(ticker=ticker)
    if not rows:
        return EvaluationReport(
            total_records=0,
            directional_records=0,
            hold_records=0,
            direction_accuracy=0.0,
            buy_accuracy=0.0,
            sell_accuracy=0.0,
            signal_counts={},
            avg_return_by_signal={},
            accuracy_by_confidence={},
            rolling_accuracy=[],
        )

    # ── Signal counts ────────────────────────────────────────────
    signal_counts: dict[str, int] = {}
    for row in rows:
        sig = row["signal"]
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

    # ── Direction accuracy ───────────────────────────────────────
    directional = [r for r in rows if r["signal"] in _DIRECTION_SIGNALS]
    holds = [r for r in rows if r["signal"] == "HOLD"]

    correct = [r for r in directional if r["direction_correct"]]
    direction_accuracy = len(correct) / len(directional) if directional else 0.0

    buys = [r for r in directional if r["signal"] == "BUY"]
    buy_correct = [r for r in buys if r["direction_correct"]]
    buy_accuracy = len(buy_correct) / len(buys) if buys else 0.0

    sells = [r for r in directional if r["signal"] == "SELL"]
    sell_correct = [r for r in sells if r["direction_correct"]]
    sell_accuracy = len(sell_correct) / len(sells) if sells else 0.0

    # ── Average return by signal ─────────────────────────────────
    avg_return_by_signal: dict[str, float] = {}
    for sig in signal_counts:
        sig_rows = [r for r in rows if r["signal"] == sig and r["actual_return"] is not None]
        if sig_rows:
            avg_return_by_signal[sig] = round(
                sum(r["actual_return"] for r in sig_rows) / len(sig_rows), 2
            )

    # ── Confidence calibration ───────────────────────────────────
    accuracy_by_confidence: dict[str, dict[str, Any]] = {}
    for level in ("HIGH", "MEDIUM", "LOW", ""):
        level_rows = [
            r for r in directional if _normalize_confidence(r["confidence"]) == level
        ]
        if level_rows:
            level_correct = sum(1 for r in level_rows if r["direction_correct"])
            label = level if level else "UNKNOWN"
            accuracy_by_confidence[label] = {
                "accuracy": round(level_correct / len(level_rows), 2),
                "count": len(level_rows),
            }

    # ── Rolling accuracy ─────────────────────────────────────────
    rolling: list[tuple[str, float]] = []
    for i in range(len(directional)):
        window_start = max(0, i - rolling_window + 1)
        window = directional[window_start : i + 1]
        window_correct = sum(1 for r in window if r["direction_correct"])
        rolling.append(
            (directional[i]["trade_date"], round(window_correct / len(window), 2))
        )

    return EvaluationReport(
        total_records=len(rows),
        directional_records=len(directional),
        hold_records=len(holds),
        direction_accuracy=round(direction_accuracy, 4),
        buy_accuracy=round(buy_accuracy, 4),
        sell_accuracy=round(sell_accuracy, 4),
        signal_counts=signal_counts,
        avg_return_by_signal=avg_return_by_signal,
        accuracy_by_confidence=accuracy_by_confidence,
        rolling_accuracy=rolling,
        records=rows,
    )


def _normalize_confidence(raw: str) -> str:
    """Extract HIGH/MEDIUM/LOW from potentially messy confidence text."""
    if not raw:
        return ""
    upper = raw.upper().strip()
    if "HIGH" in upper:
        return "HIGH"
    if "MEDIUM" in upper or "MED" in upper:
        return "MEDIUM"
    if "LOW" in upper:
        return "LOW"
    return ""


def format_report(report: EvaluationReport) -> str:
    """Format evaluation report as readable text."""
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append("TradingAgents Evaluation Report")
    lines.append("=" * 50)
    lines.append("")

    lines.append(f"Total records:          {report.total_records}")
    lines.append(f"Directional (BUY+SELL): {report.directional_records}")
    lines.append(f"HOLD:                   {report.hold_records}")
    lines.append("")

    # Direction accuracy
    lines.append("── Direction Accuracy ──")
    lines.append(f"Overall:  {report.direction_accuracy:.1%}")
    lines.append(f"BUY:      {report.buy_accuracy:.1%} ({report.signal_counts.get('BUY', 0)} signals)")
    lines.append(f"SELL:     {report.sell_accuracy:.1%} ({report.signal_counts.get('SELL', 0)} signals)")
    lines.append(f"HOLD:     {report.signal_counts.get('HOLD', 0)} signals (not scored)")
    lines.append("")

    # Average return
    lines.append("── Average Return by Signal ──")
    for sig, avg in sorted(report.avg_return_by_signal.items()):
        lines.append(f"{sig:6s}:  {avg:+.2f}%")
    lines.append("")

    # Confidence calibration
    if report.accuracy_by_confidence:
        lines.append("── Confidence Calibration ──")
        for level, data in sorted(report.accuracy_by_confidence.items()):
            lines.append(
                f"{level:8s}: {data['accuracy']:.1%} accuracy ({data['count']} decisions)"
            )
        lines.append("")

    # Rolling accuracy
    if report.rolling_accuracy:
        lines.append("── Rolling Accuracy ──")
        for date, acc in report.rolling_accuracy[-10:]:
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            lines.append(f"{date}  {bar}  {acc:.0%}")
        lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)
