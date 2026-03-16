from tradingagents.exceptions import (
    TradingAgentsError,
    DataFetchError,
    SignalProcessingError,
    VerificationError,
)


def test_base_exception():
    err = TradingAgentsError("generic")
    assert isinstance(err, Exception)


def test_data_fetch_error():
    err = DataFetchError("yfinance", "timeout")
    assert isinstance(err, TradingAgentsError)
    assert err.source == "yfinance"
    assert err.reason == "timeout"
    assert "yfinance" in str(err)


def test_signal_processing_error_truncates():
    err = SignalProcessingError("x" * 500)
    assert len(err.raw_output) == 200


def test_signal_processing_error_short():
    err = SignalProcessingError("short")
    assert err.raw_output == "short"


def test_verification_error():
    err = VerificationError("unknown_blog", 0.3)
    assert err.source == "unknown_blog"
    assert err.confidence == 0.3
    assert "0.30" in str(err)
