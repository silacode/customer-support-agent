import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path


def get_db_path() -> Path:
    """Get the database path from environment or use default."""
    db_path = os.getenv("DATABASE_PATH", "data/support.db")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection():
    """
    Context manager for database connections.

    Usage:
        with get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users")
    """
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def execute_query(query: str) -> list[dict]:
    """
    Execute a read-only SQL query and return results as list of dicts.

    Only SELECT statements are allowed for safety.
    """
    # Basic safety check - only allow SELECT queries
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed for safety reasons")

    with get_connection() as conn:
        cursor = conn.execute(query)
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
