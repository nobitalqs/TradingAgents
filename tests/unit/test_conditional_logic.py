"""Tests for conditional logic routing."""

from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic


class TestAnalystRouter:
    def setup_method(self):
        self.logic = ConditionalLogic()

    def test_routes_to_tools_on_tool_calls(self):
        router = self.logic.make_analyst_router("market")
        msg = MagicMock()
        msg.tool_calls = [{"name": "get_stock_data"}]
        state = {"messages": [msg]}
        assert router(state) == "tools_market"

    def test_routes_to_clear_on_no_tool_calls(self):
        router = self.logic.make_analyst_router("market")
        msg = MagicMock()
        msg.tool_calls = []
        state = {"messages": [msg]}
        assert router(state) == "Msg Clear Market"

    def test_routes_to_clear_on_empty_messages(self):
        router = self.logic.make_analyst_router("news")
        state = {"messages": []}
        assert router(state) == "Msg Clear News"

    def test_router_uses_correct_analyst_type(self):
        for analyst in ["market", "social", "news", "fundamentals"]:
            router = self.logic.make_analyst_router(analyst)
            state = {"messages": []}
            result = router(state)
            assert analyst.capitalize() in result


class TestDebateRouting:
    def test_continues_to_bear_after_bull(self):
        logic = ConditionalLogic(max_debate_rounds=2)
        state = {
            "investment_debate_state": {
                "count": 1,
                "current_response": "Bull Analyst: ...",
            }
        }
        assert logic.should_continue_debate(state) == "Bear Researcher"

    def test_continues_to_bull_after_bear(self):
        logic = ConditionalLogic(max_debate_rounds=2)
        state = {
            "investment_debate_state": {
                "count": 1,
                "current_response": "Bear Analyst: ...",
            }
        }
        assert logic.should_continue_debate(state) == "Bull Researcher"

    def test_ends_debate_at_max_rounds(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = {
            "investment_debate_state": {
                "count": 2,  # 2 * 1 = 2
                "current_response": "Bull",
            }
        }
        assert logic.should_continue_debate(state) == "Research Manager"


class TestRiskRouting:
    def test_aggressive_to_conservative(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = {
            "risk_debate_state": {"count": 1, "latest_speaker": "Aggressive"}
        }
        assert logic.should_continue_risk_analysis(state) == "Conservative Analyst"

    def test_conservative_to_neutral(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = {
            "risk_debate_state": {"count": 2, "latest_speaker": "Conservative"}
        }
        assert logic.should_continue_risk_analysis(state) == "Neutral Analyst"

    def test_neutral_to_aggressive(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = {
            "risk_debate_state": {"count": 3, "latest_speaker": "Neutral"}
        }
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

    def test_ends_risk_at_max_rounds(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = {
            "risk_debate_state": {"count": 3, "latest_speaker": "Neutral"}
        }
        assert logic.should_continue_risk_analysis(state) == "Risk Judge"
