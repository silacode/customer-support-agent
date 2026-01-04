#!/usr/bin/env python3
"""Customer Support Agent CLI - Interactive support chatbot with SQL and RAG tools."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.prompt import Prompt

from agent import SupportAgent
from utils import (
    console,
    print_welcome,
    show_tool_call,
    show_agent_activity,
    initialize,
)


async def main():
    """Main entry point for the CLI."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error:[/red] OPENAI_API_KEY environment variable is required."
        )
        console.print(
            "Create a .env file with your API key or export it in your shell."
        )
        sys.exit(1)

    try:
        initialize()
    except Exception as e:
        console.print(f"[red]Initialization error:[/red] {e}")
        sys.exit(1)

    agent = SupportAgent(
        on_tool_call=show_tool_call,
        on_agent_activity=show_agent_activity,
    )

    print_welcome()

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")

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

            console.print("[cyan]Thinking...[/cyan]")
            response = await agent.chat(user_input)

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
