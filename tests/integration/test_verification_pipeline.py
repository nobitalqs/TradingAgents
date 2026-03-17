"""Integration tests: DataVerifier + NewsVerifier → CredibilitySummary pipeline."""

from __future__ import annotations

import pytest

from tradingagents.verification.data_verifier import DataVerifier
from tradingagents.verification.models import CredibilitySummary, NewsCredibility


class TestVerificationPipeline:
    """End-to-end DataVerifier produces correct CredibilitySummary."""

    def setup_method(self):
        self.verifier = DataVerifier()

    # ── Price consistency ────────────────────────────────────────────

    def test_consistent_multi_source_prices(self):
        """Multiple sources within 1% → high confidence, no warnings."""
        summary = self.verifier.build_credibility_summary(
            price_data={"yfinance": 150.0, "alpha_vantage": 150.5, "iex": 149.8},
        )
        assert summary.price_confidence == 0.95
        assert summary.warnings == []

    def test_moderate_deviation_prices(self):
        """1-2% deviation → 0.7 confidence, no warnings (threshold is < 0.7)."""
        # mean=101.5, max deviation=1.5/101.5 ≈ 1.48% → within 2% band
        summary = self.verifier.build_credibility_summary(
            price_data={"yfinance": 100.0, "alpha_vantage": 103.0},
        )
        assert summary.price_confidence == 0.7
        assert not any("low confidence" in w.lower() for w in summary.warnings)

    def test_large_deviation_prices(self):
        """>2% deviation → 0.5 confidence + warning."""
        summary = self.verifier.build_credibility_summary(
            price_data={"yfinance": 100.0, "alpha_vantage": 105.0},
        )
        assert summary.price_confidence == 0.5
        assert any("low confidence" in w.lower() for w in summary.warnings)

    def test_single_source_price(self):
        summary = self.verifier.build_credibility_summary(
            price_data={"yfinance": 200.0},
        )
        assert summary.price_confidence == 0.5
        assert any("low confidence" in w.lower() for w in summary.warnings)

    def test_no_valid_prices(self):
        summary = self.verifier.build_credibility_summary(
            price_data={"yfinance": None, "alpha_vantage": None},
        )
        assert summary.price_confidence == 0.0

    # ── News credibility ─────────────────────────────────────────────

    def test_tier1_reliable_news(self):
        """Reuters T1 + clean headline → high score, is_reliable."""
        summary = self.verifier.build_credibility_summary(
            news_items=[
                {"headline": "NVDA beats earnings", "source": "Reuters"},
            ],
        )
        assert len(summary.news_items) == 1
        item = summary.news_items[0]
        assert item.source_tier == "T1"
        assert item.is_reliable
        assert item.score >= 0.8

    def test_unknown_source_suspicious_headline(self):
        """Unknown source + suspicious pattern → low score, unreliable."""
        summary = self.verifier.build_credibility_summary(
            news_items=[
                {
                    "headline": "Guaranteed profit! Act now!",
                    "source": "random-blog.xyz",
                },
            ],
        )
        assert len(summary.news_items) == 1
        item = summary.news_items[0]
        assert item.source_tier == "unknown"
        assert not item.is_reliable
        assert "guaranteed_profit" in item.flags
        assert "urgency_pressure" in item.flags

    def test_mixed_news_generates_warning(self):
        """Mix of reliable and unreliable → warning about low credibility items."""
        summary = self.verifier.build_credibility_summary(
            news_items=[
                {"headline": "Solid earnings beat", "source": "Bloomberg"},
                {"headline": "Secret tip insider exclusive", "source": "pump-dump.io"},
                {"headline": "Breaking guaranteed profit", "source": "scam.net"},
            ],
        )
        reliable = [n for n in summary.news_items if n.is_reliable]
        unreliable = [n for n in summary.news_items if not n.is_reliable]
        assert len(reliable) >= 1
        assert len(unreliable) >= 1
        assert any("low credibility" in w.lower() for w in summary.warnings)

    # ── Combined pipeline ────────────────────────────────────────────

    def test_full_pipeline_price_and_news(self):
        """Both price and news assessed together → unified summary."""
        summary = self.verifier.build_credibility_summary(
            price_data={"yfinance": 300.0, "alpha_vantage": 300.2},
            news_items=[
                {"headline": "Strong Q4 results", "source": "WSJ"},
                {"headline": "Guaranteed profit secret tip", "source": "spam.com"},
            ],
        )
        # Price
        assert summary.price_confidence == 0.95
        # News
        assert len(summary.news_items) == 2
        # At least one warning (the spam item)
        assert len(summary.warnings) >= 1

    def test_summary_serialization_roundtrip(self):
        """to_dict and to_prompt_text produce consistent, non-empty output."""
        summary = self.verifier.build_credibility_summary(
            price_data={"src_a": 100.0, "src_b": 102.5},
            news_items=[{"headline": "NVDA news", "source": "CNBC"}],
        )
        d = summary.to_dict()
        assert "price_confidence" in d
        assert isinstance(d["news_items"], list)
        assert len(d["news_items"]) == 1
        assert d["news_items"][0]["source_name"] == "CNBC"

        prompt = summary.to_prompt_text()
        assert "DATA RELIABILITY ASSESSMENT" in prompt
        assert "CNBC" in prompt

    def test_empty_inputs_returns_defaults(self):
        """No price data, no news → pristine summary."""
        summary = self.verifier.build_credibility_summary()
        assert summary.price_confidence == 1.0
        assert summary.news_items == []
        assert summary.warnings == []
