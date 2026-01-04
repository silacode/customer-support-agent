#!/usr/bin/env python3
"""Customer Support Agent CLI - Interactive support chatbot with SQL and RAG tools."""

import asyncio
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from agent import SupportAgent
from database import init_database, seed_database
from rag import PolicyStore


console = Console()


def print_welcome():
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


def initialize():
    """Initialize database and vector store."""
    with console.status("[cyan]Initializing database...[/cyan]"):
        init_database()
        seed_database()

    with console.status("[cyan]Loading policy documents...[/cyan]"):
        store = PolicyStore()
        if store.is_empty():
            count = store.load_documents()
            console.print(f"[dim]Loaded {count} policy document chunks[/dim]")

    console.print("[green]✓[/green] System initialized\n")


def show_tool_call(name: str, args: dict) -> None:
    """Display tool calls as they happen."""
    if name == "query_orders_database":
        query = args.get("query", "")
        display = f"{query[:80]}..." if len(query) > 80 else query
        console.print("  [dim]⚡ Querying database...[/dim]")
        console.print(f"     [dim italic]{display}[/dim italic]")
    elif name == "search_policies":
        console.print(f"  [dim]📋 Searching policies: {args.get('question', '')}[/dim]")


async def main():
    """Main entry point for the CLI."""
    # Load environment variables
    load_dotenv()

    # Check for API key
    import os

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error:[/red] OPENAI_API_KEY environment variable is required."
        )
        console.print(
            "Create a .env file with your API key or export it in your shell."
        )
        sys.exit(1)

    # Initialize systems
    try:
        initialize()
    except Exception as e:
        console.print(f"[red]Initialization error:[/red] {e}")
        sys.exit(1)

    # Create agent with tool call display callback
    agent = SupportAgent(on_tool_call=show_tool_call)

    # Print welcome
    print_welcome()

    # Main REPL loop
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")

            # Handle commands
            command = user_input.strip().lower()
            if command in ("quit", "exit", "q"):
                console.print("\n[cyan]Goodbye![/cyan]")
                break
            elif command == "clear":
                agent.clear_history()
                console.print("[dim]Conversation history cleared.[/dim]\n")
                continue
            elif command == "help":
                print_welcome()
                continue
            elif not user_input.strip():
                continue

            # Get agent response
            console.print("[cyan]Thinking...[/cyan]")
            response = await agent.chat(user_input)

            # Display response
            console.print()
            console.print("[bold blue]Agent[/bold blue]")
            console.print(Markdown(response))
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
