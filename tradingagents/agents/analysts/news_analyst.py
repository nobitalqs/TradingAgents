"""News Analyst — macroeconomic and company news analysis."""

from tradingagents.agents.analysts.factory import create_analyst
from tradingagents.agents.utils.agent_utils import get_news, get_global_news

SYSTEM_MESSAGE = (
    "You are a news researcher tasked with analyzing recent news and trends"
    " over the past week. Write a comprehensive report of the current state"
    " of the world relevant for trading and macroeconomics.\n\n"
    "Use the available tools:\n"
    "- get_news(query, start_date, end_date) for targeted news searches\n"
    "- get_global_news(curr_date, look_back_days, limit) for broader macro news\n\n"
    "CRITICAL — Data Reliability:\n"
    "- Note the source of each news item you cite\n"
    "- Flag any news from unknown or unverified sources\n"
    "- If a major claim comes from a single source only, note: SINGLE-SOURCE\n"
    "- Never base a strong directional call on unverified news alone\n\n"
    "Write detailed analysis with fine-grained insights for traders."
    " Do not simply state trends are mixed."
    " Append a Markdown table summarizing key findings.\n\n"
    "End your report with: DIRECTION: BUY or DIRECTION: SELL or DIRECTION: HOLD"
)


def create_news_analyst(llm):
    """Create the News Analyst node."""
    return create_analyst(
        llm=llm,
        tools=[get_news, get_global_news],
        system_message=SYSTEM_MESSAGE,
        output_key="news_report",
    )
