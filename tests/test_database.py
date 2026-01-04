"""Tests for database layer: connection, models, and seed."""

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from database.connection import execute_query, get_connection, get_db_path
from database.models import init_database
from database.seed import seed_database


class TestGetDbPath:
    """Tests for get_db_path function."""

    def test_returns_default_path(self, tmp_path: Path):
        """Should return default path when DATABASE_PATH not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear DATABASE_PATH if it exists
            os.environ.pop("DATABASE_PATH", None)
            path = get_db_path()
            assert path == Path("data/support.db")

    def test_returns_env_path(self, tmp_path: Path):
        """Should return path from DATABASE_PATH env var."""
        custom_path = tmp_path / "custom.db"
        with patch.dict(os.environ, {"DATABASE_PATH": str(custom_path)}):
            path = get_db_path()
            assert path == custom_path

    def test_creates_parent_directory(self, tmp_path: Path):
        """Should create parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "dir" / "test.db"
        with patch.dict(os.environ, {"DATABASE_PATH": str(nested_path)}):
            path = get_db_path()
            assert path.parent.exists()


class TestGetConnection:
    """Tests for get_connection context manager."""

    def test_returns_connection(self, temp_db_path: Path):
        """Should return a valid SQLite connection."""
        with get_connection() as conn:
            assert conn is not None
            cursor = conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1

    def test_connection_has_row_factory(self, temp_db_path: Path):
        """Connection should have Row factory for dict-like access."""
        with get_connection() as conn:
            assert conn.row_factory == sqlite3.Row

    def test_closes_connection_on_exit(self, temp_db_path: Path):
        """Connection should be closed after context exits."""
        with get_connection() as conn:
            pass
        # Attempting to use closed connection should fail
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestExecuteQuery:
    """Tests for execute_query function."""

    def test_executes_select_query(self, seeded_db: Path):
        """Should execute SELECT queries and return results."""
        results = execute_query("SELECT name, email FROM customers LIMIT 2")
        assert len(results) == 2
        assert "name" in results[0]
        assert "email" in results[0]

    def test_returns_list_of_dicts(self, seeded_db: Path):
        """Results should be a list of dictionaries."""
        results = execute_query("SELECT id, name FROM customers LIMIT 1")
        assert isinstance(results, list)
        assert isinstance(results[0], dict)

    def test_rejects_insert_query(self, initialized_db: Path):
        """Should reject INSERT queries."""
        with pytest.raises(ValueError, match="Only SELECT queries"):
            execute_query("INSERT INTO customers (name, email) VALUES ('Test', 'test@example.com')")

    def test_rejects_update_query(self, initialized_db: Path):
        """Should reject UPDATE queries."""
        with pytest.raises(ValueError, match="Only SELECT queries"):
            execute_query("UPDATE customers SET name = 'Test'")

    def test_rejects_delete_query(self, initialized_db: Path):
        """Should reject DELETE queries."""
        with pytest.raises(ValueError, match="Only SELECT queries"):
            execute_query("DELETE FROM customers")

    def test_rejects_drop_query(self, initialized_db: Path):
        """Should reject DROP queries."""
        with pytest.raises(ValueError, match="Only SELECT queries"):
            execute_query("DROP TABLE customers")

    def test_empty_result(self, seeded_db: Path):
        """Should return empty list when no rows match."""
        results = execute_query("SELECT * FROM customers WHERE email = 'nonexistent@example.com'")
        assert results == []


class TestInitDatabase:
    """Tests for init_database function."""

    def test_creates_tables(self, temp_db_path: Path):
        """Should create all required tables."""
        init_database()

        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

        # Filter out sqlite internal tables
        tables = [t for t in tables if not t.startswith("sqlite_")]
        expected_tables = ["customers", "order_items", "orders", "products"]
        assert sorted(tables) == expected_tables

    def test_idempotent(self, temp_db_path: Path):
        """Should be safe to call multiple times."""
        init_database()
        init_database()  # Should not raise

        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM customers")
            assert cursor.fetchone()[0] == 0


class TestSeedDatabase:
    """Tests for seed_database function."""

    def test_seeds_customers(self, initialized_db: Path):
        """Should seed customer data."""
        seed_database()

        results = execute_query("SELECT COUNT(*) as count FROM customers")
        assert results[0]["count"] == 5

    def test_seeds_products(self, initialized_db: Path):
        """Should seed product data."""
        seed_database()

        results = execute_query("SELECT COUNT(*) as count FROM products")
        assert results[0]["count"] == 8

    def test_seeds_orders(self, initialized_db: Path):
        """Should seed order data."""
        seed_database()

        results = execute_query("SELECT COUNT(*) as count FROM orders")
        assert results[0]["count"] == 6

    def test_seeds_order_items(self, initialized_db: Path):
        """Should seed order items data."""
        seed_database()

        results = execute_query("SELECT COUNT(*) as count FROM order_items")
        assert results[0]["count"] == 8

    def test_idempotent(self, initialized_db: Path):
        """Should only seed once (idempotent)."""
        seed_database()
        seed_database()  # Should not add more data

        results = execute_query("SELECT COUNT(*) as count FROM customers")
        assert results[0]["count"] == 5

