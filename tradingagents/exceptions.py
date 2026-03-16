"""Custom exception hierarchy for TradingAgents."""


class TradingAgentsError(Exception):
    """Base exception for all TradingAgents errors."""


class DataFetchError(TradingAgentsError):
    """Core data unavailable — terminates analysis."""

    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(f"Data fetch failed from {source}: {reason}")


class SignalProcessingError(TradingAgentsError):
    """Signal extraction failed after all fallback levels."""

    def __init__(self, raw_output: str):
        self.raw_output = raw_output[:200]
        super().__init__(f"Failed to extract signal from: {self.raw_output}")


class DecisionExtractionError(TradingAgentsError):
    """Decision extraction from LLM output failed."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Decision extraction failed: {reason}")


class VerificationError(TradingAgentsError):
    """Data verification failed — credibility below threshold."""

    def __init__(self, source: str, confidence: float):
        self.source = source
        self.confidence = confidence
        super().__init__(
            f"Verification failed for {source}: confidence={confidence:.2f}"
        )
