"""4-level signal extraction for trading decisions.

Levels:
    L1 - Regex extraction from raw text
    L2 - LLM extraction followed by regex validation
    L3 - Strict LLM (must output exactly BUY/SELL/HOLD)
    L4 - Raise SignalProcessingError
"""

import logging
import re

from tradingagents.exceptions import SignalProcessingError

logger = logging.getLogger("tradingagents.graph.signal")

_SIGNAL_RE = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)

_EXTRACTION_PROMPT = (
    "You are an efficient assistant designed to analyze paragraphs or "
    "financial reports provided by a group of analysts. Your task is to "
    "extract the investment decision: SELL, BUY, or HOLD. Provide only "
    "the extracted decision (SELL, BUY, or HOLD) as your output, without "
    "adding any additional text or information."
)

_STRICT_PROMPT = (
    "Output exactly one word: BUY, SELL, or HOLD. "
    "No other text, punctuation, or explanation."
)


def extract_signal(text: str) -> str | None:
    """Extract trading signal via regex. Returns LAST match (uppercase) or None."""
    matches = _SIGNAL_RE.findall(text)
    return matches[-1].upper() if matches else None


class SignalProcessor:
    """Processes trading signals through a 4-level extraction pipeline."""

    def __init__(self, llm=None):
        """Initialize with an optional LLM for L2/L3 extraction.

        Args:
            llm: A LangChain-compatible chat model (e.g. ChatOpenAI).
                 If None, only L1 regex extraction is available.
        """
        self.llm = llm

    def process_signal(self, full_signal: str) -> str:
        """Process a full trading signal through all extraction levels.

        Args:
            full_signal: Complete trading signal text.

        Returns:
            Extracted decision: "BUY", "SELL", or "HOLD".

        Raises:
            SignalProcessingError: If all levels fail to extract a valid signal.
        """
        # L1: Direct regex extraction
        result = self._level1_regex(full_signal)
        if result is not None:
            logger.debug("L1 regex extracted signal: %s", result)
            return result

        # L2: LLM extraction + regex validation
        result = self._level2_llm_regex(full_signal)
        if result is not None:
            logger.debug("L2 LLM+regex extracted signal: %s", result)
            return result

        # L3: Strict LLM (output exactly BUY/SELL/HOLD)
        result = self._level3_strict_llm(full_signal)
        if result is not None:
            logger.debug("L3 strict LLM extracted signal: %s", result)
            return result

        # L4: All levels exhausted
        logger.error("All signal extraction levels failed for input: %.200s", full_signal)
        raise SignalProcessingError(
            f"Failed to extract signal from text after all levels: {full_signal[:200]}"
        )

    def _level1_regex(self, text: str) -> str | None:
        """L1: Extract signal directly via regex."""
        return extract_signal(text)

    def _level2_llm_regex(self, text: str) -> str | None:
        """L2: Use LLM to extract, then validate with regex."""
        if self.llm is None:
            logger.debug("L2 skipped: no LLM configured")
            return None

        try:
            messages = [
                ("system", _EXTRACTION_PROMPT),
                ("human", text),
            ]
            response = self.llm.invoke(messages).content
            logger.debug("L2 LLM response: %s", response)
            return extract_signal(response)
        except Exception:
            logger.warning("L2 LLM invocation failed", exc_info=True)
            return None

    def _level3_strict_llm(self, text: str) -> str | None:
        """L3: Strict LLM that must output exactly BUY/SELL/HOLD."""
        if self.llm is None:
            logger.debug("L3 skipped: no LLM configured")
            return None

        try:
            messages = [
                ("system", _STRICT_PROMPT),
                ("human", text),
            ]
            response = self.llm.invoke(messages).content.strip().upper()
            logger.debug("L3 strict LLM response: %s", response)
            if response in ("BUY", "SELL", "HOLD"):
                return response
            return None
        except Exception:
            logger.warning("L3 strict LLM invocation failed", exc_info=True)
            return None
