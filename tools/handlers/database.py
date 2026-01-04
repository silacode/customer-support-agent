"""Database query tool handler with SQL reflection pattern."""

import json
from typing import Callable

from database import execute_query
from database.models import SCHEMA
from agent.sql_generator import SQLGeneratorAgent
from agent.sql_reviewer import SQLReviewerAgent


MAX_RETRIES = 4

# Callback type for agent activity notifications
AgentCallback = Callable[[str, str, dict], None]


async def query_orders_database(
    query: str,
    on_agent_activity: AgentCallback | None = None,
) -> str:
    """
    Execute a SQL query against the orders database using the reflection pattern.

    Uses SQLGeneratorAgent to generate SQL and SQLReviewerAgent to validate,
    with up to MAX_RETRIES attempts if the query needs correction.

    Args:
        query: Natural language question to answer
        on_agent_activity: Optional callback for agent activity notifications

    Returns:
        JSON formatted query results or error message
    """
    generator = SQLGeneratorAgent(on_activity=on_agent_activity)
    reviewer = SQLReviewerAgent(on_activity=on_agent_activity)
    feedback: str | None = None
    last_result: str = "No results found."

    for attempt in range(MAX_RETRIES):
        try:
            # Generate SQL query
            raw_sql = await generator.generate(query, SCHEMA, feedback)

            # Notify about SQL execution
            if on_agent_activity:
                on_agent_activity(
                    "Database",
                    "executing",
                    {"sql": raw_sql, "attempt": attempt + 1},
                )

            # Execute the query
            results = execute_query(raw_sql)

            if not results:
                result_str = "No results found."
            else:
                result_str = json.dumps(results, indent=2, default=str)

            last_result = result_str

            # Review the query and results
            feedback = await reviewer.review(query, SCHEMA, raw_sql, result_str)

            # If no feedback, the query is correct
            if feedback is None:
                return result_str

        except ValueError as e:
            # Query validation error (e.g., not a SELECT query)
            feedback = f"Query error: {str(e)}. Please generate a valid SELECT query."
        except Exception as e:
            # Database execution error
            feedback = f"Database error: {str(e)}. Please fix the SQL syntax."

    # Return the last result after max retries
    return last_result
