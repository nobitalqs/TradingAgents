"""Tests for analyst factory pattern."""

from unittest.mock import MagicMock

from tradingagents.agents.analysts.factory import create_analyst


def _make_mock_llm(content="Test report", tool_calls=None):
    """Create a mock LLM compatible with LangChain pipe operator.

    When piped with ChatPromptTemplate, the mock gets wrapped in
    RunnableLambda which calls it as mock(input) rather than mock.invoke(input).
    So we need to set both .return_value and .invoke.return_value.
    """
    mock_llm = MagicMock()
    result = MagicMock()
    result.content = content
    result.tool_calls = tool_calls or []

    bound = MagicMock()
    bound.return_value = result        # for RunnableLambda(bound)(input)
    bound.invoke.return_value = result  # for bound.invoke(input)

    mock_llm.bind_tools.return_value = bound
    return mock_llm, bound, result


def test_factory_returns_callable():
    llm, _, _ = _make_mock_llm()
    node_fn = create_analyst(
        llm=llm, tools=[], system_message="Test", output_key="test_report",
    )
    assert callable(node_fn)


def test_factory_writes_to_output_key():
    llm, _, _ = _make_mock_llm(content="My analysis report")
    node_fn = create_analyst(
        llm=llm, tools=[], system_message="Analyze", output_key="market_report",
    )

    state = {
        "messages": [],
        "trade_date": "2026-01-15",
        "company_of_interest": "NVDA",
    }

    result = node_fn(state)
    assert result["market_report"] == "My analysis report"
    assert "messages" in result


def test_factory_empty_report_on_tool_calls():
    """When LLM returns tool calls, report should be empty (tool loop continues)."""
    llm, _, _ = _make_mock_llm(
        content="some content", tool_calls=[{"name": "get_stock_data", "args": {}}]
    )

    node_fn = create_analyst(
        llm=llm, tools=[], system_message="Test", output_key="test_report",
    )

    state = {
        "messages": [],
        "trade_date": "2026-01-15",
        "company_of_interest": "AAPL",
    }

    result = node_fn(state)
    assert result["test_report"] == ""


def test_factory_invokes_bound_llm():
    """Verify the LLM's bind_tools was called and the bound LLM was invoked."""
    llm, bound, _ = _make_mock_llm()

    node_fn = create_analyst(
        llm=llm, tools=[], system_message="Check", output_key="report",
    )

    state = {
        "messages": [("human", "analyze")],
        "trade_date": "2026-03-15",
        "company_of_interest": "TSLA",
    }

    node_fn(state)
    assert llm.bind_tools.called
    assert bound.called  # RunnableLambda calls bound(prompt_output)


def test_each_analyst_creator_works():
    """Smoke test that each analyst creator returns a callable."""
    from tradingagents.agents.analysts.market_analyst import create_market_analyst
    from tradingagents.agents.analysts.news_analyst import create_news_analyst
    from tradingagents.agents.analysts.social_media_analyst import (
        create_social_media_analyst,
    )
    from tradingagents.agents.analysts.fundamentals_analyst import (
        create_fundamentals_analyst,
    )

    llm, _, _ = _make_mock_llm()

    for creator in [
        create_market_analyst,
        create_news_analyst,
        create_social_media_analyst,
        create_fundamentals_analyst,
    ]:
        node = creator(llm)
        assert callable(node)
