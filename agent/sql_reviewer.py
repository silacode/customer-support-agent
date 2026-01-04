"""SQL Reviewer agent for validating SQL queries using LLM-as-judge pattern."""

import os
from typing import Callable

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputItemParam, EasyInputMessageParam


# Callback type for agent activity notifications
AgentCallback = Callable[[str, str, dict], None]

INSTRUCTIONS = """You are a SQL query reviewer.

Think step by step:
1. Understand what the original question is asking for
2. Analyze if the SQL query logic matches the question intent
3. Verify the query results actually answer the question
4. Check for common issues: wrong JOINs, missing WHERE clauses, incorrect aggregations

Output:
- If correct: respond with exactly "CORRECT"
- If incorrect: provide specific, actionable feedback on what's wrong and how to fix it

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
