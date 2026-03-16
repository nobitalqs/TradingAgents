"""Centralized constants — eliminate magic strings."""

ANALYST_TYPES = ("market", "social", "news", "fundamentals")
DEFAULT_ANALYST_ORDER = list(ANALYST_TYPES)

VALID_SIGNALS = ("BUY", "SELL", "HOLD")


def analyst_node_name(analyst_type: str) -> str:
    """Generate graph node name for an analyst."""
    return f"{analyst_type.capitalize()} Analyst"


def msg_clear_node_name(analyst_type: str) -> str:
    """Generate graph node name for message clearing."""
    return f"Msg Clear {analyst_type.capitalize()}"


def tools_node_name(analyst_type: str) -> str:
    """Generate graph node name for tool execution."""
    return f"tools_{analyst_type}"
