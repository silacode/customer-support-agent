"""CLI display utilities."""

from rich.console import Console
from rich.panel import Panel


console = Console()


def print_welcome() -> None:
    """Print welcome message and available commands."""
    console.print(
        Panel.fit(
            "[bold cyan]Customer Support Agent[/bold cyan]\n\n"
            "Ask me about orders, products, stock levels, or company policies.\n\n"
            "[dim]Commands:[/dim]\n"
            "  [green]quit[/green] or [green]exit[/green] - Exit the agent\n"
            "  [green]clear[/green] - Clear conversation history\n"
            "  [green]help[/green] - Show this message",
            title="Welcome",
            border_style="cyan",
        )
    )
    console.print()


def show_tool_call(name: str, args: dict) -> None:
    """Display tool calls as they happen."""
    if name == "query_orders_database":
        query = args.get("query", "")
        display = f"{query[:80]}..." if len(query) > 80 else query
        console.print("  [dim]⚡ Tool: query_orders_database[/dim]")
        console.print(f"     [dim italic]{display}[/dim italic]")
    elif name == "search_policies":
        console.print("  [dim]📋 Tool: search_policies[/dim]")
        console.print(f"     [dim italic]{args.get('question', '')}[/dim italic]")


def show_agent_activity(agent_name: str, action: str, details: dict) -> None:
    """Display agent activity as it happens."""
    attempt = details.get("attempt", 1)
    retry_info = f" (retry {attempt})" if attempt > 1 else ""
    console.print(f"     [dim]→ {agent_name}: {action}{retry_info}[/dim]")
