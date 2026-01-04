"""Policy search tool handler."""

from rag import PolicyStore


# Global policy store instance (initialized lazily)
_policy_store: PolicyStore | None = None


def get_policy_store() -> PolicyStore:
    """Get or initialize the policy store."""
    global _policy_store
    if _policy_store is None:
        _policy_store = PolicyStore()
        if _policy_store.is_empty():
            _policy_store.load_documents()
    return _policy_store


async def search_policies(question: str) -> str:
    """
    Search company policies for relevant information.

    Args:
        question: The policy-related question to search for

    Returns:
        Formatted policy results or error message
    """
    store = get_policy_store()
    results = store.search(question, n_results=3)

    if not results:
        return "No relevant policies found."

    formatted = []
    for doc in results:
        formatted.append(
            f"**{doc['title']}** (distance: {doc['distance']:.2f})\n{doc['content']}"
        )

    return "\n\n---\n\n".join(formatted)
