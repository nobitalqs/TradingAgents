"""Structured decision extraction from LLM trading signals.

Parses LLM output into a validated TradingDecision dataclass with
confidence overrides from analyst consensus.
"""

import json
import logging
import re
from dataclasses import dataclass

from tradingagents.exceptions import DecisionExtractionError
from tradingagents.graph.signal_processing import SignalProcessor

logger = logging.getLogger("tradingagents.graph.decision_extraction")

_VALID_SIGNALS = frozenset({"BUY", "SELL", "HOLD"})
_VALID_CONFIDENCES = frozenset({"HIGH", "MEDIUM", "LOW"})

_DECISION_PROMPT = (
    "Analyze the following trading signal and output a JSON object with exactly "
    "these keys:\n"
    '- "signal": one of "BUY", "SELL", "HOLD"\n'
    '- "confidence": one of "HIGH", "MEDIUM", "LOW"\n'
    '- "position_pct": a float between 0.0 and 1.0 (fraction of portfolio)\n'
    '- "reasoning": a brief explanation\n\n'
    "Output ONLY valid JSON, no other text."
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class TradingDecision:
    """A validated, structured trading decision."""

    signal: str  # "BUY" | "SELL" | "HOLD"
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    position_pct: float  # 0.0 to 1.0
    reasoning: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def _validate_and_build(
    signal: str,
    confidence: str,
    position_pct: float,
    reasoning: str,
) -> TradingDecision:
    """Validate fields and build a TradingDecision.

    Args:
        signal: Trading direction.
        confidence: Confidence level.
        position_pct: Portfolio fraction.
        reasoning: Explanation text.

    Returns:
        A validated TradingDecision.

    Raises:
        DecisionExtractionError: If signal or confidence is invalid.
    """
    signal_upper = signal.strip().upper()
    if signal_upper not in _VALID_SIGNALS:
        raise DecisionExtractionError(f"Invalid signal: {signal!r}")

    confidence_upper = confidence.strip().upper()
    if confidence_upper not in _VALID_CONFIDENCES:
        raise DecisionExtractionError(f"Invalid confidence: {confidence!r}")

    clamped_pct = _clamp(float(position_pct), 0.0, 1.0)

    return TradingDecision(
        signal=signal_upper,
        confidence=confidence_upper,
        position_pct=clamped_pct,
        reasoning=str(reasoning).strip(),
    )


class DecisionExtractor:
    """Extracts structured trading decisions from LLM output."""

    def __init__(self, llm):
        """Initialize with a LangChain-compatible LLM.

        Args:
            llm: Chat model for structured extraction.
        """
        self.llm = llm
        self._signal_processor = SignalProcessor(llm)

    def extract(
        self,
        full_signal: str,
        analyst_consensus: dict | None = None,
    ) -> TradingDecision:
        """Extract a structured decision from LLM output.

        Strategy:
            1. Try to parse JSON directly from full_signal
            2. Ask LLM to produce structured JSON
            3. Fall back to SignalProcessor for signal + defaults

        Applies post-processing:
            - Override confidence from consensus if available
            - Reduce position_pct to max 0.3 for LOW confidence BUY

        Args:
            full_signal: Raw trading signal text.
            analyst_consensus: Optional consensus dict with 'confidence' key.

        Returns:
            A validated TradingDecision.

        Raises:
            DecisionExtractionError: If extraction fails completely.
        """
        decision = self._try_json_parse(full_signal)
        if decision is None:
            decision = self._try_llm_extraction(full_signal)
        if decision is None:
            decision = self._try_signal_fallback(full_signal)

        if decision is None:
            raise DecisionExtractionError(
                f"Failed to extract decision from: {full_signal[:200]}"
            )

        # Post-processing: override confidence from consensus
        decision = self._apply_consensus_override(decision, analyst_consensus)

        # Post-processing: reduce position for low-confidence buys
        decision = self._apply_position_cap(decision)

        return decision

    def _try_json_parse(self, text: str) -> TradingDecision | None:
        """Try to extract a JSON object directly from text."""
        match = _JSON_RE.search(text)
        if not match:
            return None

        try:
            data = json.loads(match.group())
            return _validate_and_build(
                signal=data["signal"],
                confidence=data["confidence"],
                position_pct=data["position_pct"],
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.debug("JSON parse failed: %s", exc)
            return None

    def _try_llm_extraction(self, text: str) -> TradingDecision | None:
        """Ask LLM to produce structured JSON output."""
        try:
            messages = [
                ("system", _DECISION_PROMPT),
                ("human", text),
            ]
            response = self.llm.invoke(messages).content
            logger.debug("LLM extraction response: %s", response)
            return self._try_json_parse(response)
        except Exception:
            logger.warning("LLM extraction failed", exc_info=True)
            return None

    def _try_signal_fallback(self, text: str) -> TradingDecision | None:
        """Fall back to signal processor + default values."""
        try:
            signal = self._signal_processor.process_signal(text)
            return TradingDecision(
                signal=signal,
                confidence="LOW",
                position_pct=0.5,
                reasoning="Extracted via signal processor fallback",
            )
        except Exception:
            logger.warning("Signal fallback failed", exc_info=True)
            return None

    def _apply_consensus_override(
        self,
        decision: TradingDecision,
        analyst_consensus: dict | None,
    ) -> TradingDecision:
        """Override confidence from analyst consensus if available."""
        if analyst_consensus is None:
            return decision

        consensus_confidence = analyst_consensus.get("confidence", "").strip().upper()
        if consensus_confidence in _VALID_CONFIDENCES:
            if consensus_confidence != decision.confidence:
                logger.info(
                    "Overriding confidence %s -> %s from consensus",
                    decision.confidence,
                    consensus_confidence,
                )
                return TradingDecision(
                    signal=decision.signal,
                    confidence=consensus_confidence,
                    position_pct=decision.position_pct,
                    reasoning=decision.reasoning,
                )

        return decision

    def _apply_position_cap(self, decision: TradingDecision) -> TradingDecision:
        """Cap position_pct to 0.3 for low-confidence BUY signals."""
        if decision.confidence == "LOW" and decision.signal == "BUY":
            capped_pct = min(decision.position_pct, 0.3)
            if capped_pct != decision.position_pct:
                logger.info(
                    "Capping position_pct %.2f -> %.2f for LOW confidence BUY",
                    decision.position_pct,
                    capped_pct,
                )
                return TradingDecision(
                    signal=decision.signal,
                    confidence=decision.confidence,
                    position_pct=capped_pct,
                    reasoning=decision.reasoning,
                )

        return decision
