"""Pytest fixtures shared by every test module."""

import os
from collections.abc import Iterator

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.retrieval.chunking.chunk_store import Document
from scene_rag.retrieval.chunking.settings import CHUNK_STORE_URI_ENV
from tests.support.chunking_fakes import ChunkingHarness, make_chunking_harness


@pytest.fixture
def chunking_harness() -> ChunkingHarness:
    """A chunking service wired to fresh fakes over an empty scene-api.

    Tests seed scene sets with `chunking_harness.seed(...)`; a class may override this
    fixture, requesting it by the same name, to seed state before its cases run.
    """
    return make_chunking_harness()


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
