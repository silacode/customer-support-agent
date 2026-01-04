import asyncio
import os
import json
from typing import Any, Callable, cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputItemParam, EasyInputMessageParam

from tools import TOOLS, handle_tool_call

# Callback type for tool call notifications
ToolCallCallback = Callable[[str, dict], None]

# Callback type for agent activity notifications
AgentCallback = Callable[[str, str, dict], None]


INSTRUCTIONS = """You are a helpful customer support agent for an e-commerce company. 
Your role is to assist customers with:

1. **Order inquiries**: Check order status, tracking information, and order history
2. **Product questions**: Stock availability, product details, and pricing
3. **Policy questions**: Returns, shipping, warranties, and other company policies

You have access to two tools:
- **query_orders_database**: Query the orders database for customer, order, and product information
- **search_policies**: Search company policies for procedural questions

Guidelines:
- Be friendly, professional, and concise
- Always verify information using the available tools before responding
- If you can't find the information, let the customer know and offer alternatives
- For order lookups, you may need to search by customer email or order ID
- Provide specific, actionable answers based on the data you retrieve

Remember: You can only use SELECT queries on the database. The available tables are:
- customers (id, name, email, phone)
- products (id, name, description, price, stock_quantity, category)  
- orders (id, customer_id, status, total_amount, shipping_address, tracking_number, created_at)
- order_items (id, order_id, product_id, quantity, unit_price)"""


class SupportAgent:
    """Customer support agent using OpenAI Responses API with tool calling."""

    def __init__(
        self,
        model: str | None = None,
        on_tool_call: ToolCallCallback | None = None,
        on_agent_activity: AgentCallback | None = None,
    ):
        """
        Initialize the support agent.

        Args:
            model: OpenAI model to use. Defaults to OPENAI_MODEL env var or gpt-5-mini.
            on_tool_call: Optional callback invoked when a tool is called.
                          Receives (tool_name, arguments_dict).
            on_agent_activity: Optional callback for sub-agent activity notifications.
                               Receives (agent_name, action, details_dict).
        """
        self.client = AsyncOpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.conversation: list[ResponseInputItemParam] = []
        self.on_tool_call = on_tool_call
        self.on_agent_activity = on_agent_activity

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            user_message: The user's input message

        Returns:
            The agent's response
        """
        user_msg: EasyInputMessageParam = {"role": "user", "content": user_message}
        self.conversation.append(user_msg)

        # Call OpenAI Responses API
        response = await self.client.responses.create(
            model=self.model,
            instructions=INSTRUCTIONS,
            input=self.conversation,
            tools=TOOLS,
            reasoning={"effort": "medium"},
        )

        # Process output items - handle function calls
        while self._has_function_calls(response.output):
            # Execute function calls and collect results
            tool_outputs = await self._process_function_calls(response.output)

            # Add outputs to conversation and get next response
            self.conversation.extend(
                cast(list[ResponseInputItemParam], response.output)
            )
            self.conversation.extend(tool_outputs)

            response = await self.client.responses.create(
                model=self.model,
                instructions=INSTRUCTIONS,
                input=self.conversation,
                tools=TOOLS,
                reasoning={"effort": "medium"},
            )

        # Add final response to conversation history
        self.conversation.extend(cast(list[ResponseInputItemParam], response.output))

        return response.output_text

    def _has_function_calls(self, output: list[Any]) -> bool:
        """Check if output contains any function calls."""
        return any(item.type == "function_call" for item in output)

    async def _process_function_calls(
        self, output: list[Any]
    ) -> list[ResponseInputItemParam]:
        """Execute function calls and return output items."""
        tasks = []
        call_ids = []

        for item in output:
            if item.type == "function_call":
                call_ids.append(item.call_id)
                tasks.append(self._execute_tool(item.name, item.arguments))

        results = await asyncio.gather(*tasks)

        return cast(
            list[ResponseInputItemParam],
            [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                }
                for call_id, result in zip(call_ids, results)
            ],
        )

    async def _execute_tool(self, name: str, arguments: str) -> str:
        """Execute a single tool call and return the result."""
        args = json.loads(arguments)

        if self.on_tool_call:
            self.on_tool_call(name, args)

        return await handle_tool_call(name, args, self.on_agent_activity)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation = []
