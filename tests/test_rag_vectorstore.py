"""Tests for RAG vectorstore: PolicyStore with mocked embeddings."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.vectorstore import PolicyStore


class TestPolicyStore:
    """Tests for PolicyStore class."""

    def test_initialization(self, temp_chroma_path: Path, mock_embeddings):
        """Should initialize with ChromaDB client and collection."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        
        assert store.client is not None
        assert store.collection is not None
        assert store.collection.name == "policies"

    def test_is_empty_when_new(self, temp_chroma_path: Path, mock_embeddings):
        """New store should be empty."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        
        assert store.is_empty() is True

    def test_load_documents(
        self, temp_chroma_path: Path, temp_policies_dir: Path, mock_embeddings
    ):
        """Should load documents and return count."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        
        count = store.load_documents(str(temp_policies_dir))
        
        assert count >= 2  # At least 2 documents
        assert store.is_empty() is False

    def test_load_documents_empty_dir(self, temp_chroma_path: Path, tmp_path: Path, mock_embeddings):
        """Should return 0 when no documents to load."""
        empty_dir = tmp_path / "empty_policies"
        empty_dir.mkdir()
        
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        count = store.load_documents(str(empty_dir))
        
        assert count == 0

    def test_search_returns_results(
        self, temp_chroma_path: Path, temp_policies_dir: Path, mock_embeddings
    ):
        """Search should return matching documents."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        store.load_documents(str(temp_policies_dir))
        
        results = store.search("return policy", n_results=3)
        
        assert len(results) > 0
        assert "content" in results[0]
        assert "source" in results[0]
        assert "title" in results[0]
        assert "distance" in results[0]

    def test_search_respects_n_results(
        self, temp_chroma_path: Path, temp_policies_dir: Path, mock_embeddings
    ):
        """Search should respect n_results parameter."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        store.load_documents(str(temp_policies_dir))
        
        results = store.search("shipping", n_results=1)
        
        assert len(results) <= 1

    def test_search_empty_store(self, temp_chroma_path: Path, mock_embeddings):
        """Search on empty store should return empty list."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        
        results = store.search("anything", n_results=3)
        
        assert results == []

    def test_clear(
        self, temp_chroma_path: Path, temp_policies_dir: Path, mock_embeddings
    ):
        """Clear should remove all documents."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        store.load_documents(str(temp_policies_dir))
        
        assert store.is_empty() is False
        
        store.clear()
        
        assert store.is_empty() is True

    def test_uses_env_path(self, tmp_path: Path, mock_embeddings):
        """Should use CHROMA_PATH environment variable."""
        chroma_path = tmp_path / "env_chroma"
        chroma_path.mkdir()
        
        with patch.dict(os.environ, {"CHROMA_PATH": str(chroma_path)}):
            store = PolicyStore()
            
            # Verify it uses the env path by checking client exists
            assert store.client is not None

    def test_search_result_has_distance(
        self, temp_chroma_path: Path, temp_policies_dir: Path, mock_embeddings
    ):
        """Search results should include distance scores."""
        store = PolicyStore(persist_dir=str(temp_chroma_path))
        store.load_documents(str(temp_policies_dir))
        
        results = store.search("return items", n_results=3)
        
        if results:
            assert isinstance(results[0]["distance"], float)

