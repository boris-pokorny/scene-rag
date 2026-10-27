"""Pytest fixtures shared by every test module."""

import os
from collections.abc import Iterator

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.extraction.ingestion.registry import Document
from scene_rag.extraction.ingestion.settings import REGISTRY_URI_ENV


@pytest.fixture(scope="session")
def mongo_uri() -> str:
    """Connection string of the MongoDB integration tests use."""
    return os.environ.get(REGISTRY_URI_ENV, "mongodb://localhost:27017")


@pytest.fixture(scope="session")
def mongo_client(mongo_uri: str) -> Iterator[MongoClient[Document]]:
    """A client for the MongoDB named by the environment; skips the test when none answers."""
    uri = mongo_uri
    client: MongoClient[Document] = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=500)
    try:
        client.admin.command("ping")
    except PyMongoError:
        pytest.skip(f"no MongoDB at {uri}")
    yield client
    client.close()
