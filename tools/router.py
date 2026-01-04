"""Tool routing and execution."""

from typing import Callable, Awaitable

from .handlers import query_orders_database, search_policies


# Callback type for agent activity notifications
AgentCallback = Callable[[str, str, dict], None]

# Registry mapping tool names to handler functions
TOOL_HANDLERS: dict[str, Callable[..., Awaitable[str]]] = {
    "query_orders_database": query_orders_database,
    "search_policies": search_policies,
}

# Tools that support agent activity callbacks
TOOLS_WITH_AGENT_CALLBACK = {"query_orders_database"}


async def handle_tool_call(
    name: str,
    arguments: dict,
    on_agent_activity: AgentCallback | None = None,
) -> str:
    """
    Route tool calls to their respective handlers.

    Args:
        name: Name of the tool to execute
        arguments: Arguments for the tool
        on_agent_activity: Optional callback for agent activity notifications

    Returns:
        Result from the tool execution

    Raises:
        ValueError: If the tool name is not recognized
    """
    if name not in TOOL_HANDLERS:
        raise ValueError(f"Unknown tool: {name}")

    try:
        handler = TOOL_HANDLERS[name]
        if name in TOOLS_WITH_AGENT_CALLBACK:
            return await handler(**arguments, on_agent_activity=on_agent_activity)
        return await handler(**arguments)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"
