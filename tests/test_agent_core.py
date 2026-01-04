"""Tests for the main SupportAgent."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai import APIConnectionError
from openai.types.responses import EasyInputMessageParam


class MockOutputItem:
    """Mock for response output items."""

    def __init__(self, item_type: str, **kwargs):
        self.type = item_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockResponse:
    """Mock for OpenAI Responses API response."""

    def __init__(self, output_text: str = "", output: list | None = None):
        self.output_text = output_text
        self.output = output or [MockOutputItem("message", content=output_text)]


class TestSupportAgent:
    """Tests for SupportAgent class."""

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """Should return agent response text."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MockResponse(output_text="Hello! How can I help you?")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.core import SupportAgent

            agent = SupportAgent()
            result = await agent.chat("Hello")

            assert result == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_maintains_conversation_history(self):
        """Should maintain conversation history across calls."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MockResponse(output_text="Response")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.core import SupportAgent

            agent = SupportAgent()
            await agent.chat("First message")
            await agent.chat("Second message")

            # Conversation should have 4 items: 2 user messages + 2 responses
            assert len(agent.conversation) == 4

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Should clear conversation history."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MockResponse(output_text="Response")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.core import SupportAgent

            agent = SupportAgent()
            await agent.chat("Hello")

            agent.clear_history()

            assert len(agent.conversation) == 0

    @pytest.mark.asyncio
    async def test_handles_function_calls(self):
        """Should handle function call responses."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            with patch("agent.core.handle_tool_call") as mock_handle:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                # First response has function call
                function_call_output = MockOutputItem(
                    "function_call",
                    call_id="call_123",
                    name="search_policies",
                    arguments='{"question": "return policy"}',
                )
                first_response = MockResponse(
                    output_text="", output=[function_call_output]
                )

                # Second response is final message
                final_response = MockResponse(
                    output_text="Based on the policy, you can return within 30 days."
                )

                mock_client.responses.create = AsyncMock(
                    side_effect=[first_response, final_response]
                )
                mock_handle.return_value = "Return policy: 30 days"

                from agent.core import SupportAgent

                agent = SupportAgent()
                result = await agent.chat("What's the return policy?")

                assert result == "Based on the policy, you can return within 30 days."
                mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_invokes_tool_callback(self):
        """Should invoke tool call callback when set."""
        callback_calls = []

        def mock_callback(name, args):
            callback_calls.append((name, args))

        with patch("agent.core.AsyncOpenAI") as mock_openai:
            with patch("agent.core.handle_tool_call") as mock_handle:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                function_call_output = MockOutputItem(
                    "function_call",
                    call_id="call_123",
                    name="search_policies",
                    arguments='{"question": "test"}',
                )
                first_response = MockResponse(
                    output_text="", output=[function_call_output]
                )
                final_response = MockResponse(output_text="Done")

                mock_client.responses.create = AsyncMock(
                    side_effect=[first_response, final_response]
                )
                mock_handle.return_value = "result"

                from agent.core import SupportAgent

                agent = SupportAgent(on_tool_call=mock_callback)
                await agent.chat("Test")

                assert len(callback_calls) == 1
                assert callback_calls[0][0] == "search_policies"
                assert callback_calls[0][1] == {"question": "test"}

    @pytest.mark.asyncio
    async def test_multiple_function_calls(self):
        """Should handle multiple function calls in single response."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            with patch("agent.core.handle_tool_call") as mock_handle:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                # Response with two function calls
                call1 = MockOutputItem(
                    "function_call",
                    call_id="call_1",
                    name="search_policies",
                    arguments='{"question": "returns"}',
                )
                call2 = MockOutputItem(
                    "function_call",
                    call_id="call_2",
                    name="query_orders_database",
                    arguments='{"query": "SELECT 1"}',
                )
                first_response = MockResponse(output_text="", output=[call1, call2])
                final_response = MockResponse(output_text="Here's the info")

                mock_client.responses.create = AsyncMock(
                    side_effect=[first_response, final_response]
                )
                mock_handle.side_effect = ["policy result", "db result"]

                from agent.core import SupportAgent

                agent = SupportAgent()
                result = await agent.chat("Get order and policy info")

                assert result == "Here's the info"
                assert mock_handle.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_custom_model(self):
        """Should use custom model when specified."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MockResponse(output_text="Response")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            from agent.core import SupportAgent

            agent = SupportAgent(model="gpt-4")
            await agent.chat("Hello")

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4"

    def test_has_function_calls_true(self):
        """_has_function_calls should return True when output has function calls."""
        with patch("agent.core.AsyncOpenAI"):
            from agent.core import SupportAgent

            agent = SupportAgent()
            output = [MockOutputItem("function_call", call_id="1", name="test")]

            assert agent._has_function_calls(output) is True

    def test_has_function_calls_false(self):
        """_has_function_calls should return False when no function calls."""
        with patch("agent.core.AsyncOpenAI"):
            from agent.core import SupportAgent

            agent = SupportAgent()
            output = [MockOutputItem("message", content="text")]

            assert agent._has_function_calls(output) is False

    def test_has_function_calls_mixed(self):
        """_has_function_calls should return True with mixed output."""
        with patch("agent.core.AsyncOpenAI"):
            from agent.core import SupportAgent

            agent = SupportAgent()
            output = [
                MockOutputItem("message", content="text"),
                MockOutputItem("function_call", call_id="1", name="test"),
            ]

            assert agent._has_function_calls(output) is True

    def test_truncate_conversation_when_over_limit(self):
        """Should truncate conversation when over MAX_CONVERSATION_ITEMS."""
        with patch("agent.core.AsyncOpenAI"):
            from agent.core import SupportAgent, MAX_CONVERSATION_ITEMS

            agent = SupportAgent()
            # Add more items than the limit
            for i in range(MAX_CONVERSATION_ITEMS + 10):
                msg: EasyInputMessageParam = {"role": "user", "content": f"msg {i}"}
                agent.conversation.append(msg)

            agent._truncate_conversation()

            assert len(agent.conversation) == MAX_CONVERSATION_ITEMS

    def test_truncate_conversation_keeps_recent(self):
        """Should keep most recent messages when truncating."""
        with patch("agent.core.AsyncOpenAI"):
            from agent.core import SupportAgent, MAX_CONVERSATION_ITEMS

            agent = SupportAgent()
            # Add numbered messages
            for i in range(MAX_CONVERSATION_ITEMS + 5):
                msg: EasyInputMessageParam = {"role": "user", "content": f"msg {i}"}
                agent.conversation.append(msg)

            agent._truncate_conversation()

            # Should have kept the last MAX_CONVERSATION_ITEMS messages
            first_item = cast(EasyInputMessageParam, agent.conversation[0])
            last_item = cast(EasyInputMessageParam, agent.conversation[-1])
            assert first_item.get("content") == "msg 5"
            assert last_item.get("content") == f"msg {MAX_CONVERSATION_ITEMS + 4}"

    def test_truncate_conversation_no_op_when_under_limit(self):
        """Should not truncate when under limit."""
        with patch("agent.core.AsyncOpenAI"):
            from agent.core import SupportAgent

            agent = SupportAgent()
            msg1: EasyInputMessageParam = {"role": "user", "content": "msg 1"}
            msg2: EasyInputMessageParam = {"role": "user", "content": "msg 2"}
            agent.conversation.append(msg1)
            agent.conversation.append(msg2)

            agent._truncate_conversation()

            assert len(agent.conversation) == 2

    @pytest.mark.asyncio
    async def test_tool_timeout_returns_error(self):
        """Should return error message when tool times out."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            with patch("agent.core.handle_tool_call") as mock_handle:
                with patch("agent.core.TOOL_TIMEOUT_SECONDS", 0.01):
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client

                    # Make handle_tool_call hang
                    async def slow_handler(*args, **kwargs):
                        await asyncio.sleep(1)
                        return "result"

                    mock_handle.side_effect = slow_handler

                    from agent.core import SupportAgent

                    agent = SupportAgent()
                    result = await agent._execute_tool(
                        "search_policies", '{"question": "test"}'
                    )

                    assert "timed out" in result

    @pytest.mark.asyncio
    async def test_api_retry_on_transient_error(self):
        """Should retry API calls on transient errors."""
        with patch("agent.core.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Fail twice, then succeed
            mock_response = MockResponse(output_text="Success after retry")
            mock_client.responses.create = AsyncMock(
                side_effect=[
                    APIConnectionError(request=MagicMock()),
                    APIConnectionError(request=MagicMock()),
                    mock_response,
                ]
            )

            from agent.core import SupportAgent

            agent = SupportAgent()
            result = await agent.chat("Hello")

            assert result == "Success after retry"
            assert mock_client.responses.create.call_count == 3
