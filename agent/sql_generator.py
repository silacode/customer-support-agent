"""SQL Generator agent for generating SQL queries from natural language."""

import os
from typing import Callable

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputItemParam, EasyInputMessageParam


# Callback type for agent activity notifications
AgentCallback = Callable[[str, str, dict], None]

INSTRUCTIONS = """You are a SQL query generator for SQLite.

Think step by step:
1. Analyze the user's question to understand what data they need
2. Identify which tables and columns are relevant from the schema
3. Determine the appropriate JOINs, WHERE clauses, and aggregations
4. Generate the correct SQL query

Rules:
- Only SELECT queries allowed (no INSERT, UPDATE, DELETE)
- Output ONLY the raw SQL query - no explanations, no markdown, no code blocks
- Use exact table and column names from the schema
- If feedback is provided, analyze what went wrong and fix it

Database Schema:
{schema}"""


class SQLGeneratorAgent:
    """Agent that generates SQL queries from natural language questions."""

    def __init__(
        self,
        model: str | None = None,
        on_activity: AgentCallback | None = None,
    ):
        """Initialize the SQL generator agent."""
        self.client = AsyncOpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.on_activity = on_activity

    async def generate(
        self, question: str, schema: str, feedback: str | None = None
    ) -> str:
        """
        Generate a SQL query for the given question.

        Args:
            question: The natural language question to answer
            schema: The database schema
            feedback: Optional feedback from a previous attempt

        Returns:
            A raw SQL SELECT query string
        """
        # Notify about agent activity
        if self.on_activity:
            self.on_activity(
                "SQLGeneratorAgent",
                "generating",
                {"question": question, "has_feedback": feedback is not None},
            )

        input_messages: list[ResponseInputItemParam] = []

        user_msg: EasyInputMessageParam = {
            "role": "user",
            "content": f"Question: {question}",
        }
        input_messages.append(user_msg)

        if feedback:
            feedback_msg: EasyInputMessageParam = {
                "role": "user",
                "content": f"Your previous query was incorrect. Feedback: {feedback}\n\nPlease generate a corrected SQL query.",
            }
            input_messages.append(feedback_msg)

        response = await self.client.responses.create(
            model=self.model,
            instructions=INSTRUCTIONS.format(schema=schema),
            input=input_messages,
            reasoning={"effort": "medium"},
        )

        return response.output_text.strip()
