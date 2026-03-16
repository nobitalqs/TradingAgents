"""Tests for the data verification layer."""

from __future__ import annotations

import pytest

from tradingagents.verification.models import (
    CredibilitySummary,
    NewsCredibility,
    VerifiedDataPoint,
)
from tradingagents.verification.news_verifier import NewsVerifier
from tradingagents.verification.data_verifier import DataVerifier


# ---------------------------------------------------------------------------
# TestVerifiedDataPoint
# ---------------------------------------------------------------------------


class TestVerifiedDataPoint:
    def test_creation(self):
        dp = VerifiedDataPoint(
            value=150.0,
            sources=["yfinance", "polygon"],
            confidence=0.95,
            discrepancies=[],
        )
        assert dp.value == 150.0
        assert dp.sources == ["yfinance", "polygon"]
        assert dp.confidence == 0.95
        assert dp.discrepancies == []

    def test_frozen(self):
        dp = VerifiedDataPoint(
            value=150.0,
            sources=["yfinance"],
            confidence=0.9,
            discrepancies=[],
        )
        with pytest.raises(AttributeError):
            dp.value = 200.0  # type: ignore[misc]

    def test_creation_with_discrepancies(self):
        dp = VerifiedDataPoint(
            value=150.0,
            sources=["a", "b"],
            confidence=0.7,
            discrepancies=["5% deviation between a and b"],
        )
        assert len(dp.discrepancies) == 1


# ---------------------------------------------------------------------------
# TestNewsCredibility
# ---------------------------------------------------------------------------


class TestNewsCredibility:
    def test_reliable(self):
        nc = NewsCredibility(
            score=0.9,
            flags=[],
            source_tier="T1",
            headline="Fed raises rates",
            source_name="reuters",
        )
        assert nc.is_reliable is True

    def test_unreliable(self):
        nc = NewsCredibility(
            score=0.3,
            flags=["suspicious_pattern"],
            source_tier="unknown",
            headline="Secret tip!",
            source_name="randomsite.com",
        )
        assert nc.is_reliable is False

    def test_boundary_reliable(self):
        nc = NewsCredibility(
            score=0.5,
            flags=[],
            source_tier="T2",
        )
        assert nc.is_reliable is True

    def test_boundary_unreliable(self):
        nc = NewsCredibility(
            score=0.49,
            flags=[],
            source_tier="T2",
        )
        assert nc.is_reliable is False

    def test_defaults(self):
        nc = NewsCredibility(
            score=0.8,
            flags=[],
            source_tier="T1",
        )
        assert nc.headline == ""
        assert nc.source_name == ""


# ---------------------------------------------------------------------------
# TestCredibilitySummary
# ---------------------------------------------------------------------------


class TestCredibilitySummary:
    def test_to_prompt_text_format(self):
        news = [
            NewsCredibility(
                score=0.9,
                flags=[],
                source_tier="T1",
                headline="Fed raises rates",
                source_name="reuters",
            ),
            NewsCredibility(
                score=0.3,
                flags=["urgency_pressure"],
                source_tier="unknown",
                headline="ACT NOW: guaranteed profits",
                source_name="shadysite.com",
            ),
        ]
        summary = CredibilitySummary(
            price_confidence=0.95,
            news_items=news,
            warnings=["Low credibility news detected"],
        )
        text = summary.to_prompt_text()
        assert "DATA RELIABILITY ASSESSMENT" in text
        assert "Price confidence: 0.95" in text
        assert "reuters" in text.lower() or "Reuters" in text
        assert "Low credibility news detected" in text

    def test_to_dict(self):
        news = [
            NewsCredibility(
                score=0.9,
                flags=[],
                source_tier="T1",
                headline="Market update",
                source_name="bloomberg",
            ),
        ]
        summary = CredibilitySummary(
            price_confidence=0.95,
            news_items=news,
            warnings=[],
        )
        d = summary.to_dict()
        assert d["price_confidence"] == 0.95
        assert len(d["news_items"]) == 1
        assert d["news_items"][0]["score"] == 0.9
        assert d["warnings"] == []

    def test_default_price_confidence(self):
        summary = CredibilitySummary(
            news_items=[],
            warnings=[],
        )
        assert summary.price_confidence == 1.0

    def test_to_prompt_text_empty(self):
        summary = CredibilitySummary(
            news_items=[],
            warnings=[],
        )
        text = summary.to_prompt_text()
        assert "DATA RELIABILITY ASSESSMENT" in text
        assert "Price confidence: 1.0" in text


# ---------------------------------------------------------------------------
# TestNewsVerifier
# ---------------------------------------------------------------------------


class TestNewsVerifier:
    def setup_method(self):
        self.verifier = NewsVerifier()

    def test_t1_source_high_score(self):
        result = self.verifier.assess("Fed raises rates", "Reuters")
        assert result.source_tier == "T1"
        assert result.score == pytest.approx(0.9, abs=0.01)
        assert result.is_reliable is True

    def test_t1_bloomberg(self):
        result = self.verifier.assess("Earnings beat", "Bloomberg")
        assert result.source_tier == "T1"
        assert result.score == pytest.approx(0.9, abs=0.01)

    def test_t2_source_moderate_score(self):
        result = self.verifier.assess("Stock rallies", "CNBC")
        assert result.source_tier == "T2"
        assert result.score == pytest.approx(0.8, abs=0.01)

    def test_t2_yahoo_finance(self):
        result = self.verifier.assess("Market dips", "Yahoo Finance")
        assert result.source_tier == "T2"
        assert result.score == pytest.approx(0.8, abs=0.01)

    def test_unknown_source_low_score(self):
        result = self.verifier.assess("Big move coming", "randomsite.com")
        assert result.source_tier == "unknown"
        assert result.score == pytest.approx(0.4, abs=0.01)

    def test_suspicious_pattern_guaranteed_profit(self):
        result = self.verifier.assess("Guaranteed profit on this stock!", "CNBC")
        assert len(result.flags) >= 1
        assert result.score < 0.8  # reduced from T2 baseline

    def test_suspicious_pattern_insider_exclusive(self):
        result = self.verifier.assess("Insider exclusive: next big thing", "Reuters")
        assert len(result.flags) >= 1
        assert result.score < 0.9

    def test_suspicious_pattern_secret_tip(self):
        result = self.verifier.assess("Secret tip: buy now!", "randomsite.com")
        assert len(result.flags) >= 1
        # unknown (-0.3) + at least one pattern (-0.15)
        assert result.score <= 0.25 + 0.01

    def test_multiple_suspicious_patterns(self):
        result = self.verifier.assess(
            "GUARANTEED PROFIT! Secret tip insider exclusive ACT NOW!",
            "randomsite.com",
        )
        assert len(result.flags) >= 3
        # Should be clamped to 0.0
        assert result.score == pytest.approx(0.0, abs=0.01)

    def test_body_checked_for_patterns(self):
        result = self.verifier.assess(
            "Normal headline",
            "Reuters",
            body="This is a guaranteed profit opportunity",
        )
        assert len(result.flags) >= 1
        assert result.score < 0.9

    def test_assess_batch(self):
        items = [
            ("Fed raises rates", "Reuters"),
            ("Stock rallies", "CNBC"),
            ("Secret tip!", "randomsite.com"),
        ]
        results = self.verifier.assess_batch(items)
        assert len(results) == 3
        assert results[0].source_tier == "T1"
        assert results[1].source_tier == "T2"
        assert results[2].source_tier == "unknown"

    def test_score_clamped_to_zero(self):
        # Many patterns + unknown source should clamp to 0
        result = self.verifier.assess(
            "GUARANTEED PROFIT secret tip insider exclusive ACT NOW breaking",
            "randomsite.com",
        )
        assert result.score >= 0.0

    def test_score_clamped_to_one(self):
        # T1 with no patterns should not exceed 1.0
        result = self.verifier.assess("Normal news", "Reuters")
        assert result.score <= 1.0

    def test_case_insensitive_source_matching(self):
        result = self.verifier.assess("News", "REUTERS")
        assert result.source_tier == "T1"

        result = self.verifier.assess("News", "cnbc")
        assert result.source_tier == "T2"


# ---------------------------------------------------------------------------
# TestDataVerifier
# ---------------------------------------------------------------------------


class TestDataVerifier:
    def setup_method(self):
        self.verifier = DataVerifier()

    def test_consistent_prices(self):
        prices = {"yfinance": 150.0, "polygon": 150.5}
        result = self.verifier._assess_price_consistency(prices)
        # Deviation ~0.33%, should be high confidence
        assert result.confidence >= 0.9
        assert len(result.discrepancies) == 0

    def test_divergent_prices(self):
        prices = {"yfinance": 150.0, "polygon": 160.0}
        result = self.verifier._assess_price_consistency(prices)
        # Deviation ~6.5%, should be low confidence
        assert result.confidence <= 0.5
        assert len(result.discrepancies) >= 1

    def test_moderate_divergence(self):
        prices = {"yfinance": 100.0, "polygon": 103.0}
        result = self.verifier._assess_price_consistency(prices)
        # Max deviation from mean ≈ 1.48%, between 1% and 2% → 0.7
        assert result.confidence == pytest.approx(0.7, abs=0.01)

    def test_single_source(self):
        prices = {"yfinance": 150.0}
        result = self.verifier._assess_price_consistency(prices)
        assert result.confidence == 0.5
        assert len(result.sources) == 1

    def test_no_data(self):
        prices: dict[str, float | None] = {}
        result = self.verifier._assess_price_consistency(prices)
        assert result.confidence == 0.0

    def test_all_none_values(self):
        prices: dict[str, float | None] = {"yfinance": None, "polygon": None}
        result = self.verifier._assess_price_consistency(prices)
        assert result.confidence == 0.0

    def test_some_none_values(self):
        prices: dict[str, float | None] = {
            "yfinance": 150.0,
            "polygon": None,
        }
        result = self.verifier._assess_price_consistency(prices)
        # Only one valid source
        assert result.confidence == 0.5

    def test_assess_news(self):
        items = [
            {"headline": "Fed raises rates", "source": "Reuters"},
            {"headline": "Stock rallies", "source": "CNBC"},
        ]
        results = self.verifier.assess_news(items)
        assert len(results) == 2
        assert results[0].source_tier == "T1"
        assert results[1].source_tier == "T2"

    def test_assess_news_empty(self):
        results = self.verifier.assess_news([])
        assert results == []

    def test_build_summary(self):
        price_data = {"yfinance": 150.0, "polygon": 150.5}
        news_items = [
            {"headline": "Fed raises rates", "source": "Reuters"},
        ]
        summary = self.verifier.build_credibility_summary(
            price_data=price_data, news_items=news_items
        )
        assert isinstance(summary, CredibilitySummary)
        assert summary.price_confidence >= 0.9
        assert len(summary.news_items) == 1

    def test_build_summary_no_data(self):
        summary = self.verifier.build_credibility_summary()
        assert isinstance(summary, CredibilitySummary)
        assert summary.price_confidence == 1.0
        assert summary.news_items == []
        assert summary.warnings == []

    def test_build_summary_with_warnings(self):
        news_items = [
            {"headline": "Secret tip!", "source": "randomsite.com"},
        ]
        summary = self.verifier.build_credibility_summary(news_items=news_items)
        assert len(summary.warnings) >= 1

    def test_config_stored(self):
        config = {"some_key": "some_value"}
        verifier = DataVerifier(config=config)
        assert verifier.config == config
