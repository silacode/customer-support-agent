"""OpenAI Responses API tool definitions."""

from openai.types.responses import FunctionToolParam

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "query_orders_database",
        "description": (
            "Query the customer orders database to get information about orders, "
            "customers, products, and stock levels. Use SQL SELECT queries only. "
            "Available tables: customers (id, name, email, phone), "
            "products (id, name, description, price, stock_quantity, category), "
            "orders (id, customer_id, status, total_amount, shipping_address, tracking_number, created_at), "
            "order_items (id, order_id, product_id, quantity, unit_price)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A SQL SELECT query to execute against the database"
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_policies",
        "description": (
            "Search company policies for information about returns, shipping, warranties, "
            "and other customer service policies. Use this when customers ask about "
            "policies, procedures, or general company rules. "
            "Results include distance scores—lower is better (0.2 is better than 0.6)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The policy-related question to search for"
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
]
