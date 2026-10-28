"""Command semantics of the Chunk Store, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator
from typing import Protocol

import pytest
from pymongo import MongoClient

from scene_rag.retrieval.chunking.chunk_store import (
    CHUNK_SETS_COLLECTION,
    CHUNKS_COLLECTION,
    Document,
    MongoChunkStore,
    chunk_from_document,
    chunk_set_from_document,
    chunk_set_to_document,
    chunk_to_document,
)
from scene_rag.retrieval.chunking.protocols import ChunkStoreReader, ChunkStoreWriter
from tests.support.chunking_fakes import (
    FakeChunkStore,
    make_chunk,
    make_chunk_set,
    make_records,
    sha256_version,
)


class ChunkStore(ChunkStoreReader, ChunkStoreWriter, Protocol):
    """Both sides of the store, as every backend under test provides them."""


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def store(request: pytest.FixtureRequest) -> Iterator[ChunkStore]:
    if request.param == "fake":
        yield FakeChunkStore()
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield MongoChunkStore(client, database)
    client.drop_database(database)


FIRST = make_records(sha256_version(b"first"))
SECOND = make_records(sha256_version(b"second"))


class TestGivenEmptyStore:
    """No record stored."""

    def test_when_chunk_put_then_get_returns_it(self, store: ChunkStore) -> None:
        chunk = make_chunk(FIRST, 1)

        assert store.put_chunk(chunk) is True
        assert store.get_chunk(chunk.version) == chunk
        assert store.get_chunk("sha256:missing") is None

    def test_when_chunk_set_put_then_get_returns_it(self, store: ChunkStore) -> None:
        chunk_set = make_chunk_set(FIRST)

        assert store.put_chunk_set(chunk_set) is True
        assert store.get_chunk_set(chunk_set.version) == chunk_set
        assert store.get_chunk_set("sha256:missing") is None

    def test_when_nothing_stored_then_nothing_done(self, store: ChunkStore) -> None:
        assert store.list_done("0123456789ab") == frozenset()


class TestGivenStoredRecords:
    """Chunk sets under two fingerprints."""

    @pytest.fixture
    def stored(self, store: ChunkStore) -> None:
        store.put_chunk_set(make_chunk_set(FIRST, "aaaaaaaaaaaa"))
        store.put_chunk_set(make_chunk_set(SECOND, "aaaaaaaaaaaa"))
        store.put_chunk_set(make_chunk_set(FIRST, "bbbbbbbbbbbb"))
        store.put_chunk(make_chunk(FIRST, 0, "aaaaaaaaaaaa"))

    def test_when_listed_then_sets_of_that_fingerprint(
        self, store: ChunkStore, stored: None
    ) -> None:
        assert store.list_done("aaaaaaaaaaaa") == {
            FIRST[0].scene_set_version,
            SECOND[0].scene_set_version,
        }
        assert store.list_done("bbbbbbbbbbbb") == {FIRST[0].scene_set_version}
        assert store.list_done("cccccccccccc") == frozenset()

    def test_when_put_again_then_refused_and_unchanged(
        self, store: ChunkStore, stored: None
    ) -> None:
        original = make_chunk(FIRST, 0, "aaaaaaaaaaaa")
        rival_chunk = make_chunk(FIRST, 0, "aaaaaaaaaaaa", text="other")
        rival_set = make_chunk_set(FIRST, "aaaaaaaaaaaa", chunk_count=0)

        assert store.put_chunk(rival_chunk) is False
        assert store.put_chunk_set(rival_set) is False
        assert store.get_chunk(original.version) == original
        assert store.get_chunk_set(rival_set.version) == make_chunk_set(FIRST, "aaaaaaaaaaaa")


class TestGivenDocuments:
    """Records encoded as store documents."""

    def test_when_chunk_encoded_and_decoded_then_identical(self) -> None:
        chunk = make_chunk(FIRST, 2)

        document = chunk_to_document(chunk)

        assert document["_id"] == chunk.version
        assert document["scene_indices"] == [2]
        assert document["entitlements"] == ["public"]
        assert document["lineage"]["step"] == "chunking"
        assert chunk_from_document(document) == chunk

    def test_when_chunk_set_encoded_and_decoded_then_identical(self) -> None:
        chunk_set = make_chunk_set(FIRST)

        document = chunk_set_to_document(chunk_set)

        assert document["pins"] == [str(pin) for pin in chunk_set.pins]
        assert chunk_set_from_document(document) == chunk_set

    def test_when_collections_named_then_chunks_and_chunk_sets(self) -> None:
        assert (CHUNKS_COLLECTION, CHUNK_SETS_COLLECTION) == ("chunks", "chunk_sets")
