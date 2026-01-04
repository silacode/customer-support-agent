"""Tests for SQL generator and reviewer agents."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSQLGeneratorAgent:
    """Tests for SQLGeneratorAgent."""

    @pytest.mark.asyncio
    async def test_generates_sql_query(self):
        """Should generate SQL query from natural language."""
        with patch("agent.sql_generator.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "SELECT * FROM customers"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_generator import SQLGeneratorAgent

            agent = SQLGeneratorAgent()
            result = await agent.generate(
                "Get all customers",
                "CREATE TABLE customers (id INTEGER, name TEXT)"
            )

            assert result == "SELECT * FROM customers"

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        """Should strip whitespace from generated query."""
        with patch("agent.sql_generator.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "  SELECT 1  \n"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_generator import SQLGeneratorAgent

            agent = SQLGeneratorAgent()
            result = await agent.generate("Test", "schema")

            assert result == "SELECT 1"

    @pytest.mark.asyncio
    async def test_includes_feedback_in_prompt(self):
        """Should include feedback when provided."""
        with patch("agent.sql_generator.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "SELECT name FROM customers"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_generator import SQLGeneratorAgent

            agent = SQLGeneratorAgent()
            await agent.generate(
                "Get customer names",
                "schema",
                feedback="Previous query was wrong, use customers table"
            )

            # Verify create was called
            mock_client.responses.create.assert_called_once()
            
            # Check that input contains feedback
            call_kwargs = mock_client.responses.create.call_args.kwargs
            input_messages = call_kwargs["input"]
            
            # Should have 2 messages: question and feedback
            assert len(input_messages) == 2

    @pytest.mark.asyncio
    async def test_invokes_activity_callback(self):
        """Should invoke activity callback when provided."""
        callback_calls = []

        def mock_callback(agent_name, action, details):
            callback_calls.append((agent_name, action, details))

        with patch("agent.sql_generator.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "SELECT 1"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_generator import SQLGeneratorAgent

            agent = SQLGeneratorAgent(on_activity=mock_callback)
            await agent.generate("Test", "schema")

            assert len(callback_calls) == 1
            assert callback_calls[0][0] == "SQLGeneratorAgent"
            assert callback_calls[0][1] == "generating"

    @pytest.mark.asyncio
    async def test_uses_custom_model(self):
        """Should use custom model when specified."""
        with patch("agent.sql_generator.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "SELECT 1"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_generator import SQLGeneratorAgent

            agent = SQLGeneratorAgent(model="gpt-4")
            await agent.generate("Test", "schema")

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4"


class TestSQLReviewerAgent:
    """Tests for SQLReviewerAgent."""

    @pytest.mark.asyncio
    async def test_returns_none_for_correct(self):
        """Should return None when query is correct."""
        with patch("agent.sql_reviewer.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "CORRECT"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_reviewer import SQLReviewerAgent

            agent = SQLReviewerAgent()
            result = await agent.review(
                "Get customers",
                "schema",
                "SELECT * FROM customers",
                "[{id: 1, name: 'Alice'}]"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_correct_lowercase(self):
        """Should return None for 'correct' in any case."""
        with patch("agent.sql_reviewer.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "correct"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_reviewer import SQLReviewerAgent

            agent = SQLReviewerAgent()
            result = await agent.review("q", "s", "sql", "result")

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_feedback_for_incorrect(self):
        """Should return feedback string when query is incorrect."""
        with patch("agent.sql_reviewer.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "The query is missing a WHERE clause"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_reviewer import SQLReviewerAgent

            agent = SQLReviewerAgent()
            result = await agent.review(
                "Get specific customer",
                "schema",
                "SELECT * FROM customers",
                "[]"
            )

            assert result == "The query is missing a WHERE clause"

    @pytest.mark.asyncio
    async def test_invokes_activity_callback(self):
        """Should invoke activity callback when provided."""
        callback_calls = []

        def mock_callback(agent_name, action, details):
            callback_calls.append((agent_name, action, details))

        with patch("agent.sql_reviewer.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "CORRECT"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_reviewer import SQLReviewerAgent

            agent = SQLReviewerAgent(on_activity=mock_callback)
            await agent.review("q", "s", "SELECT 1", "result")

            assert len(callback_calls) == 1
            assert callback_calls[0][0] == "SQLReviewerAgent"
            assert callback_calls[0][1] == "reviewing"

    @pytest.mark.asyncio
    async def test_includes_sql_in_callback_details(self):
        """Should include SQL query in callback details."""
        callback_calls = []

        def mock_callback(agent_name, action, details):
            callback_calls.append((agent_name, action, details))

        with patch("agent.sql_reviewer.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_text = "CORRECT"
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.sql_reviewer import SQLReviewerAgent

            agent = SQLReviewerAgent(on_activity=mock_callback)
            await agent.review("q", "s", "SELECT * FROM orders", "result")

            assert callback_calls[0][2]["sql"] == "SELECT * FROM orders"

