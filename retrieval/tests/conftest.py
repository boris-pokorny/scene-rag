"""Pytest fixtures shared by every test module."""

import os
from collections.abc import Iterator

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.retrieval.chunking.chunk_store import Document
from scene_rag.retrieval.chunking.settings import CHUNK_STORE_URI_ENV
from scene_rag.retrieval.embedding.http_transport import TransportError, UrllibTransport
from scene_rag.retrieval.embedding.protocols import SearchIndexError
from scene_rag.retrieval.embedding.qdrant_index import QdrantSearchIndex
from scene_rag.retrieval.embedding.settings import SEARCH_INDEX_URL_ENV
from tests.support.chunking_fakes import ChunkingHarness, make_chunking_harness
from tests.support.embedding_fakes import EmbeddingHarness, make_embedding_harness
from tests.support.query_fakes import QueryHarness, make_query_harness


@pytest.fixture
def chunking_harness() -> ChunkingHarness:
    """A chunking service wired to fresh fakes over an empty scene-api.

    Tests seed scene sets with `chunking_harness.seed(...)`; a class may override this
    fixture, requesting it by the same name, to seed state before its cases run.
    """
    return make_chunking_harness()


@pytest.fixture
def embedding_harness() -> EmbeddingHarness:
    """An embedding service wired to fresh fakes over an empty Chunk Store.

    Tests seed chunk sets with `embedding_harness.seed(...)`; a class may override this
    fixture, requesting it by the same name, to seed state before its cases run.
    """
    return make_embedding_harness()


@pytest.fixture
def query_harness() -> QueryHarness:
    """The query use cases wired to fresh fakes over an empty Search Index.

    Tests index entries with `query_harness.seed(...)`; a class may override this fixture,
    requesting it by the same name, to seed state before its cases run.
    """
    return make_query_harness()


@pytest.fixture(scope="session")
def mongo_uri() -> str:
    """Connection string of the MongoDB integration tests use."""
    return os.environ.get(CHUNK_STORE_URI_ENV, "mongodb://localhost:27017")


@pytest.fixture(scope="session")
def mongo_client(mongo_uri: str) -> Iterator[MongoClient[Document]]:
    """A client for the MongoDB named by the environment; skips the test when none answers."""
    client: MongoClient[Document] = MongoClient(
        mongo_uri, tz_aware=True, serverSelectionTimeoutMS=500
    )
    try:
        client.admin.command("ping")
    except PyMongoError:
        pytest.skip(f"no MongoDB at {mongo_uri}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    """Base URL of the Qdrant the integration tests use."""
    return os.environ.get(SEARCH_INDEX_URL_ENV, "http://localhost:6333")


@pytest.fixture(scope="session")
def qdrant_index(qdrant_url: str) -> QdrantSearchIndex:
    """A backend over the Qdrant named by the environment; skips the test when none answers."""
    index = QdrantSearchIndex(qdrant_url, UrllibTransport(), timeout_seconds=5.0)
    try:
        index.version()
    except SearchIndexError, TransportError:
        pytest.skip(f"no Qdrant at {qdrant_url}")
    return index
