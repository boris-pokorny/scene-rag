"""Command semantics of the index ledger, on the fake and on MongoDB."""

import dataclasses
import uuid
from collections.abc import Iterator
from typing import Protocol

import pytest
from pymongo import MongoClient

from scene_rag.retrieval.embedding.index_store import (
    INDEX_SETS_COLLECTION,
    INDEXES_COLLECTION,
    Document,
    MongoIndexStore,
)
from scene_rag.retrieval.embedding.protocols import IndexStoreReader, IndexStoreWriter
from tests.support.embedding_fakes import (
    FakeIndexStore,
    make_chunk_views,
    make_index,
    make_index_set,
    sha256_version,
)


class IndexStore(IndexStoreReader, IndexStoreWriter, Protocol):
    """Both sides of the ledger, as every backend under test provides them."""


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def store(request: pytest.FixtureRequest) -> Iterator[IndexStore]:
    if request.param == "fake":
        yield FakeIndexStore()
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield MongoIndexStore(client, database)
    client.drop_database(database)


FIRST = make_chunk_views(sha256_version(b"first"))
SECOND = make_chunk_views(sha256_version(b"second"), count=1)


class TestGivenEmptyStore:
    """No record stored."""

    def test_when_index_set_put_then_get_returns_it(self, store: IndexStore) -> None:
        index_set = make_index_set(FIRST)

        assert store.put_index_set(index_set) is True
        assert store.get_index_set(index_set.version) == index_set
        assert store.get_index_set("sha256:missing") is None

    def test_when_index_put_then_get_returns_it(self, store: IndexStore) -> None:
        index = make_index()

        assert store.put_index(index) is True
        assert store.get_index(index.fingerprint) == index
        assert store.get_index("000000000000") is None

    def test_when_nothing_stored_then_nothing_done(self, store: IndexStore) -> None:
        assert store.list_done("0123456789ab") == frozenset()


class TestGivenStoredRecords:
    """Index sets under two fingerprints and one index record."""

    @pytest.fixture
    def stored(self, store: IndexStore) -> None:
        store.put_index_set(make_index_set(FIRST, fingerprint="aaaaaaaaaaaa"))
        store.put_index_set(make_index_set(SECOND, fingerprint="aaaaaaaaaaaa"))
        store.put_index_set(make_index_set(FIRST, fingerprint="bbbbbbbbbbbb"))
        store.put_index(make_index("aaaaaaaaaaaa"))

    def test_when_listed_then_sets_of_that_fingerprint(
        self, store: IndexStore, stored: None
    ) -> None:
        assert store.list_done("aaaaaaaaaaaa") == {
            make_index_set(FIRST).chunk_set_version,
            make_index_set(SECOND).chunk_set_version,
        }
        assert store.list_done("bbbbbbbbbbbb") == {make_index_set(FIRST).chunk_set_version}
        assert store.list_done("cccccccccccc") == frozenset()

    def test_when_put_again_then_refused_and_unchanged(
        self, store: IndexStore, stored: None
    ) -> None:
        original = make_index_set(FIRST, fingerprint="aaaaaaaaaaaa")
        rival_set = make_index_set(FIRST, fingerprint="aaaaaaaaaaaa", vector_count=0)
        rival_index = dataclasses.replace(make_index("aaaaaaaaaaaa"), dimensions=4)

        assert store.put_index_set(rival_set) is False
        assert store.put_index(rival_index) is False
        assert store.get_index_set(original.version) == original
        assert store.get_index("aaaaaaaaaaaa") == make_index("aaaaaaaaaaaa")


class TestGivenCollections:
    """The collection names."""

    def test_when_named_then_index_sets_and_indexes(self) -> None:
        assert (INDEX_SETS_COLLECTION, INDEXES_COLLECTION) == ("index_sets", "indexes")
