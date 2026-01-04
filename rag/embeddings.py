"""OpenAI embeddings wrapper."""

from openai import OpenAI
from chromadb.base_types import PyVector


EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(texts: list[str]) -> list[PyVector]:
    """
    Get embeddings for a list of texts using OpenAI's embedding API.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors compatible with ChromaDB
    """
    client = OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

    # Sort by index - order is not guaranteed by the API
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def get_embedding(text: str) -> PyVector:
    """Get embedding for a single text."""
    return get_embeddings([text])[0]
