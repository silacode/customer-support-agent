"""Tests for tool handlers: database and policies."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQueryOrdersDatabase:
    """Tests for query_orders_database handler."""

    @pytest.mark.asyncio
    async def test_returns_json_results(self, seeded_db: Path):
        """Should return JSON formatted query results."""
        # Mock the SQL generator and reviewer
        with patch("tools.handlers.database.SQLGeneratorAgent") as mock_gen_cls:
            with patch("tools.handlers.database.SQLReviewerAgent") as mock_rev_cls:
                mock_generator = MagicMock()
                mock_generator.generate = AsyncMock(
                    return_value="SELECT name FROM customers LIMIT 1"
                )
                mock_gen_cls.return_value = mock_generator

                mock_reviewer = MagicMock()
                mock_reviewer.review = AsyncMock(return_value=None)  # CORRECT
                mock_rev_cls.return_value = mock_reviewer

                from tools.handlers.database import query_orders_database

                result = await query_orders_database("Get customer names")

                assert "Alice Johnson" in result or "name" in result

    @pytest.mark.asyncio
    async def test_handles_no_results(self, seeded_db: Path):
        """Should handle queries that return no results."""
        with patch("tools.handlers.database.SQLGeneratorAgent") as mock_gen_cls:
            with patch("tools.handlers.database.SQLReviewerAgent") as mock_rev_cls:
                mock_generator = MagicMock()
                mock_generator.generate = AsyncMock(
                    return_value="SELECT * FROM customers WHERE email = 'nonexistent@example.com'"
                )
                mock_gen_cls.return_value = mock_generator

                mock_reviewer = MagicMock()
                mock_reviewer.review = AsyncMock(return_value=None)
                mock_rev_cls.return_value = mock_reviewer

                from tools.handlers.database import query_orders_database

                result = await query_orders_database("Find nonexistent user")

                assert "No results found" in result

    @pytest.mark.asyncio
    async def test_retries_on_feedback(self, seeded_db: Path):
        """Should retry when reviewer provides feedback."""
        with patch("tools.handlers.database.SQLGeneratorAgent") as mock_gen_cls:
            with patch("tools.handlers.database.SQLReviewerAgent") as mock_rev_cls:
                mock_generator = MagicMock()
                # First attempt wrong, second correct
                mock_generator.generate = AsyncMock(
                    side_effect=[
                        "SELECT * FROM wrong_table",
                        "SELECT name FROM customers LIMIT 1",
                    ]
                )
                mock_gen_cls.return_value = mock_generator

                mock_reviewer = MagicMock()
                # First review gives feedback, second approves
                mock_reviewer.review = AsyncMock(
                    side_effect=["Table 'wrong_table' doesn't exist", None]
                )
                mock_rev_cls.return_value = mock_reviewer

                from tools.handlers.database import query_orders_database

                # Patch execute_query to handle the bad first query
                with patch("tools.handlers.database.execute_query") as mock_exec:
                    mock_exec.side_effect = [
                        Exception("no such table: wrong_table"),
                        [{"name": "Alice Johnson"}],
                    ]

                    result = await query_orders_database("Get a customer name")

                    # Should have retried (at least 2 generate calls)
                    assert mock_generator.generate.call_count >= 2

    @pytest.mark.asyncio
    async def test_invokes_agent_callback(self, seeded_db: Path):
        """Should invoke agent activity callback."""
        callback_calls = []

        def mock_callback(agent_name, action, details):
            callback_calls.append((agent_name, action, details))

        with patch("tools.handlers.database.SQLGeneratorAgent") as mock_gen_cls:
            with patch("tools.handlers.database.SQLReviewerAgent") as mock_rev_cls:
                mock_generator = MagicMock()
                mock_generator.generate = AsyncMock(
                    return_value="SELECT 1"
                )
                mock_gen_cls.return_value = mock_generator

                mock_reviewer = MagicMock()
                mock_reviewer.review = AsyncMock(return_value=None)
                mock_rev_cls.return_value = mock_reviewer

                from tools.handlers.database import query_orders_database

                await query_orders_database(
                    "Test query",
                    on_agent_activity=mock_callback,
                )

                # Should have called callback for database execution
                assert any(call[0] == "Database" for call in callback_calls)


class TestSearchPolicies:
    """Tests for search_policies handler."""

    @pytest.mark.asyncio
    async def test_returns_formatted_results(
        self, temp_chroma_path: Path, temp_policies_dir: Path, mock_embeddings
    ):
        """Should return formatted policy results."""
        with patch.dict(os.environ, {"CHROMA_PATH": str(temp_chroma_path)}):
            # Reset the global store
            from tools.handlers import policies
            policies._policy_store = None

            from tools.handlers.policies import search_policies, get_policy_store

            # Initialize store with documents
            store = get_policy_store()
            store.load_documents(str(temp_policies_dir))

            result = await search_policies("return policy")

            # Should contain formatted content
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_handles_no_results(self, temp_chroma_path: Path, mock_embeddings):
        """Should handle when no policies match."""
        with patch.dict(os.environ, {"CHROMA_PATH": str(temp_chroma_path)}):
            from tools.handlers import policies
            policies._policy_store = None

            # Mock search to return empty
            with patch("tools.handlers.policies.PolicyStore") as mock_store_cls:
                mock_store = MagicMock()
                mock_store.is_empty.return_value = True
                mock_store.load_documents.return_value = 0
                mock_store.search.return_value = []
                mock_store_cls.return_value = mock_store

                from tools.handlers.policies import search_policies

                # Reset global store
                policies._policy_store = None

                result = await search_policies("obscure topic")

                assert "No relevant policies found" in result

    @pytest.mark.asyncio
    async def test_lazy_initialization(self, temp_chroma_path: Path, mock_embeddings):
        """Policy store should be lazily initialized."""
        with patch.dict(os.environ, {"CHROMA_PATH": str(temp_chroma_path)}):
            from tools.handlers import policies

            # Reset global store
            policies._policy_store = None

            from tools.handlers.policies import get_policy_store

            # First call should initialize
            store1 = get_policy_store()
            assert store1 is not None

            # Second call should return same instance
            store2 = get_policy_store()
            assert store1 is store2

