"""Command semantics of the Enrichment Store, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator
from typing import Protocol

import pytest
from pymongo import MongoClient

from scene_rag.extraction.enrichment.enrichment_store import (
    COLLECTION,
    Document,
    MongoEnrichmentStore,
    from_document,
    to_document,
)
from scene_rag.extraction.enrichment.models import SemanticSegment
from scene_rag.extraction.enrichment.protocols import EnrichmentStoreReader, EnrichmentStoreWriter
from tests.support.enrichment_fakes import (
    FakeEnrichmentStore,
    make_enrichment,
    make_scene_set,
    sha256_version,
)


class EnrichmentStore(EnrichmentStoreReader, EnrichmentStoreWriter, Protocol):
    """Both sides of the store, as every backend under test provides them."""


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def store(request: pytest.FixtureRequest) -> Iterator[EnrichmentStore]:
    if request.param == "fake":
        yield FakeEnrichmentStore()
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield MongoEnrichmentStore(client, database)
    client.drop_database(database)


FIRST = make_scene_set(sha256_version(b"first"))
SECOND = make_scene_set(sha256_version(b"second"))


class TestGivenEmptyStore:
    """No record stored."""

    def test_when_put_then_get_returns_it(self, store: EnrichmentStore) -> None:
        record = make_enrichment(FIRST, 1, "transcript")

        assert store.put(record) is True
        assert store.get(record.version) == record
        assert store.get("sha256:missing") is None

    def test_when_nothing_stored_then_nothing_done(self, store: EnrichmentStore) -> None:
        assert store.list_done("description", "0123456789ab") == frozenset()


class TestGivenStoredRecords:
    """Records of two enrichers under two fingerprints."""

    @pytest.fixture
    def stored(self, store: EnrichmentStore) -> None:
        store.put(make_enrichment(FIRST, 0, "description", "aaaaaaaaaaaa"))
        store.put(make_enrichment(FIRST, 2, "description", "aaaaaaaaaaaa"))
        store.put(make_enrichment(SECOND, 0, "description", "aaaaaaaaaaaa"))
        store.put(make_enrichment(FIRST, 0, "description", "bbbbbbbbbbbb"))
        store.put(make_enrichment(FIRST, 1, "transcript", "aaaaaaaaaaaa"))

    def test_when_listed_then_scenes_of_that_enricher_and_fingerprint(
        self, store: EnrichmentStore, stored: None
    ) -> None:
        assert store.list_done("description", "aaaaaaaaaaaa") == {
            (FIRST.version, 0),
            (FIRST.version, 2),
            (SECOND.version, 0),
        }
        assert store.list_done("description", "bbbbbbbbbbbb") == {(FIRST.version, 0)}
        assert store.list_done("transcript", "aaaaaaaaaaaa") == {(FIRST.version, 1)}

    def test_when_put_again_then_refused_and_unchanged(
        self, store: EnrichmentStore, stored: None
    ) -> None:
        original = make_enrichment(FIRST, 0, "description", "aaaaaaaaaaaa")
        rival = make_enrichment(FIRST, 0, "description", "aaaaaaaaaaaa", segments=())

        assert store.put(rival) is False
        assert store.get(original.version) == original


class TestGivenRecord:
    """A record with segments of both kinds, encoded as a store document."""

    def test_when_encoded_and_decoded_then_identical(self) -> None:
        record = make_enrichment(
            FIRST,
            2,
            "transcript",
            segments=(
                SemanticSegment(8.0, 9.5, "hello", "en", 0.75),
                SemanticSegment(9.5, 12.5, "world", None, None),
            ),
        )

        document = to_document(record)

        assert document["_id"] == record.version
        assert document["enricher"] == "transcript"
        assert document["segments"][1] == {
            "start_seconds": 9.5,
            "end_seconds": 12.5,
            "text": "world",
            "language": None,
            "confidence": None,
        }
        assert from_document(document) == record

    def test_when_collection_named_then_enrichments(self) -> None:
        assert COLLECTION == "enrichments"
