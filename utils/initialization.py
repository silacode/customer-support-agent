"""System initialization utilities."""

from database import init_database, seed_database
from rag import PolicyStore

from .cli import console


def initialize() -> None:
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
