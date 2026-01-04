"""SQL Reviewer agent for validating SQL queries using LLM-as-judge pattern."""

import os
from typing import Callable

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputItemParam, EasyInputMessageParam


# Callback type for agent activity notifications
AgentCallback = Callable[[str, str, dict], None]

INSTRUCTIONS = """You are a SQL query reviewer. Your task is to evaluate whether the generated SQL query correctly answers the user's question.

You will receive:
1. The original question
2. The database schema
3. The generated SQL query
4. The query results

Your job is to determine if the SQL query is correct and the results properly answer the question.

If the query is CORRECT and the results answer the question appropriately:
- Respond with exactly: CORRECT

If the query is INCORRECT or the results don't properly answer the question:
- Respond with specific feedback on what's wrong and how to fix it
- Be concise but clear about the issue

Database Schema:
{schema}"""


class SQLReviewerAgent:
    """Agent that reviews SQL queries and provides feedback."""

    def __init__(
        self,
        model: str | None = None,
        on_activity: AgentCallback | None = None,
    ):
        """Initialize the SQL reviewer agent."""
        self.client = AsyncOpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.on_activity = on_activity

    async def review(
        self, question: str, schema: str, raw_sql: str, sql_result: str
    ) -> str | None:
        """
        Review a SQL query and its results.

        Args:
            question: The original natural language question
            schema: The database schema
            raw_sql: The generated SQL query
            sql_result: The results from executing the query

        Returns:
            Feedback string if the query is incorrect, None if correct
        """
        # Notify about agent activity
        if self.on_activity:
            self.on_activity(
                "SQLReviewerAgent",
                "reviewing",
                {"sql": raw_sql},
            )

        review_prompt = f"""Original Question: {question}

        Generated SQL Query:
        {raw_sql}

        Query Results:
        {sql_result}

        Evaluate whether this SQL query correctly answers the original question."""

        user_msg: EasyInputMessageParam = {"role": "user", "content": review_prompt}
        input_messages: list[ResponseInputItemParam] = [user_msg]

        response = await self.client.responses.create(
            model=self.model,
            instructions=INSTRUCTIONS.format(schema=schema),
            input=input_messages,
            reasoning={"effort": "medium"},
        )

        result = response.output_text.strip()

        # If the reviewer says CORRECT, return None (no feedback needed)
        if result.upper() == "CORRECT":
            return None

        return result
