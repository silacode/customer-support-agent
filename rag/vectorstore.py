from chromadb import QueryResult
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.config import Settings

from .embeddings import get_embeddings, get_embedding
from .loader import load_policies


class SearchResult(TypedDict):
    """Type definition for a search result."""

    content: str
    source: str
    title: str
    distance: float


class PolicyStore:
    """Vector store for policy documents using ChromaDB."""

    COLLECTION_NAME = "policies"

    def __init__(self, persist_dir: str | None = None):
        """
        Initialize the policy store.

        Args:
            persist_dir: Directory to persist ChromaDB data.
                        Defaults to data/chroma or CHROMA_PATH env var.
        """
        if persist_dir is None:
            persist_dir = os.getenv("CHROMA_PATH", "data/chroma")

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def is_empty(self) -> bool:
        """Check if the collection is empty."""
        return self.collection.count() == 0

    def load_documents(self, policies_dir: str = "policies") -> int:
        """
        Load policy documents into the vector store.

        Args:
            policies_dir: Path to the policies directory

        Returns:
            Number of documents loaded
        """
        documents = load_policies(policies_dir)

        if not documents:
            return 0

        # Prepare data for ChromaDB
        ids = [f"{doc['source']}_{doc['chunk_index']}" for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [
            {
                "source": doc["source"],
                "title": doc["title"],
                "chunk_index": doc["chunk_index"],
            }
            for doc in documents
        ]

        # Get embeddings
        embeddings = get_embeddings(contents)

        # Add to collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

        return len(documents)

    def search(self, query: str, n_results: int = 3) -> list[SearchResult]:
        """
        Search for relevant policy documents.

        Args:
            query: The search query
            n_results: Number of results to return

        Returns:
            List of SearchResult with content, source, title, and distance
        """
        query_embedding = get_embedding(query)

        results: QueryResult = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Explicit validation with clear error messages
        if (
            not results["documents"]
            or not results["metadatas"]
            or not results["distances"]
        ):
            return []

        documents: list[SearchResult] = []
        for i in range(len(results["ids"][0])):
            documents.append(
                SearchResult(
                    content=str(results["documents"][0][i]),
                    source=str(results["metadatas"][0][i]["source"]),
                    title=str(results["metadatas"][0][i]["title"]),
                    distance=float(results["distances"][0][i]),
                )
            )

        return documents

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
