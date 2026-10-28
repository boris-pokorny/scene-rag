"""Pytest fixtures shared by every test module."""

import os
from collections.abc import Iterator

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.extraction.ingestion.registry import Document
from scene_rag.extraction.ingestion.settings import REGISTRY_URI_ENV
from tests.support.enrichment_fakes import EnrichmentHarness, make_enrichment_harness
from tests.support.fakes import FakeAssetRegistry, Harness, make_harness
from tests.support.segmentation_fakes import SegmentationHarness, make_segmentation_harness


@pytest.fixture
def fake_registry() -> FakeAssetRegistry:
    """An empty in-memory Asset Registry."""
    return FakeAssetRegistry()


@pytest.fixture
def harness() -> Harness:
    """An ingestion service wired to fresh fakes over an empty source.

    Tests set `harness.source.items`; a class may override this fixture, requesting it by the
    same name, to seed the registry before its cases run.
    """
    return make_harness()


@pytest.fixture
def segmentation_harness() -> SegmentationHarness:
    """A segmentation service wired to fresh fakes over an empty registry.

    Tests seed `segmentation_harness.assets.records`; a class may override this fixture,
    requesting it by the same name, to seed state before its cases run.
    """
    return make_segmentation_harness()


@pytest.fixture
def enrichment_harness() -> EnrichmentHarness:
    """An enrichment service wired to fresh fakes over empty stores.

    Tests seed scene sets with `enrichment_harness.seed(...)`; a class may override this
    fixture, requesting it by the same name, to seed state before its cases run.
    """
    return make_enrichment_harness()


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
