"""Analyst factory — DRY creation of analyst node functions.

All 4 analysts share the same structure:
  1. Build prompt with system message + tools
  2. Bind tools to LLM
  3. Invoke chain with state messages
  4. Return report in output_key

The only differences: system_message, tools, output_key.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger("tradingagents.agents.analysts")

_BASE_SYSTEM = (
    "You are a helpful AI assistant, collaborating with other assistants."
    " Use the provided tools to progress towards answering the question."
    " If you are unable to fully answer, that's OK; another assistant with"
    " different tools will help where you left off. Execute what you can to"
    " make progress."
    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL:"
    " **BUY/HOLD/SELL** or deliverable, prefix your response with"
    " FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
    " You have access to the following tools: {tool_names}.\n{system_message}"
    "For your reference, the current date is {current_date}."
    " The company we want to analyze is {ticker}"
)


def create_analyst(
    llm,
    tools: list,
    system_message: str,
    output_key: str,
) -> callable:
    """Create an analyst node function for the LangGraph.

    Args:
        llm: LangChain-compatible LLM instance.
        tools: List of @tool decorated functions to bind.
        system_message: Domain-specific prompt for this analyst.
        output_key: State key to write the report to.

    Returns:
        Node function: (state) -> dict
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", _BASE_SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ])

    def analyst_node(state: dict) -> dict:
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        bound_prompt = prompt.partial(
            system_message=system_message,
            tool_names=", ".join(tool.name for tool in tools),
            current_date=current_date,
            ticker=ticker,
        )

        chain = bound_prompt | llm.bind_tools(tools)
        result = chain.invoke({"messages": state["messages"]})

        report = ""
        if not result.tool_calls:
            report = result.content

        return {
            "messages": [result],
            output_key: report,
        }

    return analyst_node
