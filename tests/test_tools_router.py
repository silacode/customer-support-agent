"""Tests for tools router: tool routing and execution."""

from unittest.mock import AsyncMock, patch

import pytest


class TestToolHandlers:
    """Tests for TOOL_HANDLERS registry."""

    def test_has_query_orders_handler(self):
        """Should have query_orders_database handler."""
        from tools.router import TOOL_HANDLERS

        assert "query_orders_database" in TOOL_HANDLERS

    def test_has_search_policies_handler(self):
        """Should have search_policies handler."""
        from tools.router import TOOL_HANDLERS

        assert "search_policies" in TOOL_HANDLERS


class TestHandleToolCall:
    """Tests for handle_tool_call function."""

    @pytest.mark.asyncio
    async def test_routes_to_query_orders(self):
        """Should route query_orders_database to correct handler."""
        from tools.router import handle_tool_call

        with patch(
            "tools.router.TOOL_HANDLERS",
            {"query_orders_database": AsyncMock(return_value="mock result")},
        ):
            result = await handle_tool_call(
                "query_orders_database",
                {"query": "SELECT * FROM customers"},
            )

            assert result == "mock result"

    @pytest.mark.asyncio
    async def test_routes_to_search_policies(self):
        """Should route search_policies to correct handler."""
        from tools.router import handle_tool_call

        with patch(
            "tools.router.TOOL_HANDLERS",
            {"search_policies": AsyncMock(return_value="policy result")},
        ):
            result = await handle_tool_call(
                "search_policies",
                {"question": "What is the return policy?"},
            )

            assert result == "policy result"

    @pytest.mark.asyncio
    async def test_raises_for_unknown_tool(self):
        """Should raise ValueError for unknown tool names."""
        from tools.router import handle_tool_call

        with pytest.raises(ValueError, match="Unknown tool"):
            await handle_tool_call("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_returns_error_on_handler_exception(self):
        """Should return error string when handler raises."""
        from tools.router import handle_tool_call

        with patch(
            "tools.router.TOOL_HANDLERS",
            {"failing_tool": AsyncMock(side_effect=RuntimeError("Handler failed"))},
        ):
            result = await handle_tool_call("failing_tool", {})

            assert "Error executing failing_tool" in result
            assert "Handler failed" in result

    @pytest.mark.asyncio
    async def test_passes_agent_callback_to_supported_tools(self):
        """Should pass agent callback to tools that support it."""
        from tools.router import handle_tool_call

        mock_handler = AsyncMock(return_value="result")
        mock_callback = lambda name, action, details: None

        with patch("tools.router.TOOL_HANDLERS", {"query_orders_database": mock_handler}):
            with patch("tools.router.TOOLS_WITH_AGENT_CALLBACK", {"query_orders_database"}):
                await handle_tool_call(
                    "query_orders_database",
                    {"query": "SELECT 1"},
                    on_agent_activity=mock_callback,
                )

                # Verify callback was passed
                mock_handler.assert_called_once()
                call_kwargs = mock_handler.call_args.kwargs
                assert "on_agent_activity" in call_kwargs

    @pytest.mark.asyncio
    async def test_does_not_pass_callback_to_unsupported_tools(self):
        """Should not pass callback to tools that don't support it."""
        from tools.router import handle_tool_call

        mock_handler = AsyncMock(return_value="result")
        mock_callback = lambda name, action, details: None

        with patch("tools.router.TOOL_HANDLERS", {"search_policies": mock_handler}):
            with patch("tools.router.TOOLS_WITH_AGENT_CALLBACK", set()):
                await handle_tool_call(
                    "search_policies",
                    {"question": "test"},
                    on_agent_activity=mock_callback,
                )

                # Verify callback was NOT passed
                mock_handler.assert_called_once()
                call_kwargs = mock_handler.call_args.kwargs
                assert "on_agent_activity" not in call_kwargs
