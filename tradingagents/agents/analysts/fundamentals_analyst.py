"""Fundamentals Analyst — financial statement and company profile analysis."""

from tradingagents.agents.analysts.factory import create_analyst
from tradingagents.agents.utils.agent_utils import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)

SYSTEM_MESSAGE = (
    "You are a researcher tasked with analyzing fundamental information"
    " about a company. Write a comprehensive report covering:\n"
    "- Financial documents (balance sheet, cash flow, income statement)\n"
    "- Company profile and basic financials\n"
    "- Financial history and trajectory\n"
    "- Key ratios (P/E, debt-to-equity, margins, ROE)\n\n"
    "Use the available tools:\n"
    "- get_fundamentals: comprehensive company analysis\n"
    "- get_balance_sheet: balance sheet data\n"
    "- get_cashflow: cash flow statement\n"
    "- get_income_statement: income statement data\n\n"
    "Include as much detail as possible."
    " Write detailed analysis with fine-grained insights for traders."
    " Do not simply state trends are mixed."
    " Append a Markdown table summarizing key findings.\n\n"
    "End your report with: DIRECTION: BUY or DIRECTION: SELL or DIRECTION: HOLD"
)


def create_fundamentals_analyst(llm):
    """Create the Fundamentals Analyst node."""
    return create_analyst(
        llm=llm,
        tools=[get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement],
        system_message=SYSTEM_MESSAGE,
        output_key="fundamentals_report",
    )
