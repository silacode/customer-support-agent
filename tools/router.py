"""Tool routing and execution."""

from typing import Callable, Awaitable

from .handlers import query_orders_database, search_policies


# Registry mapping tool names to handler functions
TOOL_HANDLERS: dict[str, Callable[..., Awaitable[str]]] = {
    "query_orders_database": query_orders_database,
    "search_policies": search_policies,
}


async def handle_tool_call(name: str, arguments: dict) -> str:
    """
    Route tool calls to their respective handlers.
    
    Args:
        name: Name of the tool to execute
        arguments: Arguments for the tool
        
    Returns:
        Result from the tool execution
        
    Raises:
        ValueError: If the tool name is not recognized
    """
    if name not in TOOL_HANDLERS:
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        return await TOOL_HANDLERS[name](**arguments)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"

