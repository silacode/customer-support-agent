"""Database query tool handler."""

import json

from database import execute_query


async def query_orders_database(query: str) -> str:
    """
    Execute a SQL query against the orders database.
    
    Args:
        query: SQL SELECT query to execute
        
    Returns:
        JSON formatted query results or error message
    """
    try:
        results = execute_query(query)
        if not results:
            return "No results found."
        return json.dumps(results, indent=2, default=str)
    except ValueError as e:
        return f"Query error: {str(e)}"
    except Exception as e:
        return f"Database error: {str(e)}"

