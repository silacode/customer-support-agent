"""Shared pytest fixtures for customer support agent tests."""

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from database.models import init_database


# ------------------------------------------------------------------
# Database Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary database path."""
    db_path = tmp_path / "test_support.db"
    with patch.dict(os.environ, {"DATABASE_PATH": str(db_path)}):
        yield db_path


@pytest.fixture
def initialized_db(temp_db_path: Path) -> Generator[Path, None, None]:
    """Create and initialize a temporary database with schema."""
    with patch.dict(os.environ, {"DATABASE_PATH": str(temp_db_path)}):
        init_database()
        yield temp_db_path


@pytest.fixture
def seeded_db(initialized_db: Path) -> Generator[Path, None, None]:
    """Create a database with schema and seed data."""
    from database.seed import seed_database

    with patch.dict(os.environ, {"DATABASE_PATH": str(initialized_db)}):
        seed_database()
        yield initialized_db


# ------------------------------------------------------------------
# OpenAI Mock Fixtures
# ------------------------------------------------------------------


class MockResponseOutput:
    """Mock for OpenAI response output item."""

    def __init__(self, output_type: str, **kwargs):
        self.type = output_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockResponse:
    """Mock for OpenAI Responses API response."""

    def __init__(self, output_text: str = "", output: list | None = None):
        self.output_text = output_text
        self.output = output or [MockResponseOutput("message", content=output_text)]


@pytest.fixture
def mock_openai_client():
    """Mock AsyncOpenAI client for agent tests."""
    mock_client = MagicMock()
    mock_responses = MagicMock()
    mock_client.responses = mock_responses

    # Default response
    async def mock_create(**kwargs):
        return MockResponse(output_text="Mock response")

    mock_responses.create = AsyncMock(side_effect=mock_create)

    return mock_client


@pytest.fixture
def mock_openai_responses(mock_openai_client):
    """Patch AsyncOpenAI to return mock client."""
    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        yield mock_openai_client


# ------------------------------------------------------------------
# Embeddings Mock Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings API."""

    def fake_embeddings(texts: list[str]) -> list[list[float]]:
        """Return deterministic fake embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Create a simple deterministic embedding from text hash
            hash_val = hash(text)
            embedding = [(hash_val >> i) % 100 / 100.0 for i in range(384)]
            embeddings.append(embedding)
        return embeddings

    with patch("rag.embeddings.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def create_embeddings(model, input):
            texts = input if isinstance(input, list) else [input]
            embeddings = fake_embeddings(texts)

            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(index=i, embedding=emb)
                for i, emb in enumerate(embeddings)
            ]
            return mock_response

        mock_client.embeddings.create = create_embeddings
        yield mock_client


# ------------------------------------------------------------------
# ChromaDB Mock Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def temp_chroma_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary ChromaDB path."""
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    with patch.dict(os.environ, {"CHROMA_PATH": str(chroma_path)}):
        yield chroma_path


# ------------------------------------------------------------------
# Policy Files Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def temp_policies_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary policies directory with sample files."""
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()

    # Create sample policy files
    (policies_dir / "returns.md").write_text(
        "# Return Policy\n\nYou can return items within 30 days.\n\n"
        "Items must be unused and in original packaging."
    )
    (policies_dir / "shipping.md").write_text(
        "# Shipping Policy\n\nFree shipping on orders over $50.\n\n"
        "Standard shipping takes 5-7 business days."
    )

    yield policies_dir

