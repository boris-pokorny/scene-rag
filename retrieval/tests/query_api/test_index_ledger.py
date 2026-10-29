"""Queries over the index ledger, on the fake and on MongoDB (`integration`)."""

import uuid
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import Any, cast

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.index_ledger import (
    INDEXES_COLLECTION,
    Document,
    MongoIndexLedger,
    index_from_document,
)
from scene_rag.retrieval.query_api.protocols import IndexLedgerReader, IndexUnavailableError
from tests.support.query_fakes import T0, FakeIndexLedger, index_document, make_index_view

type Seed = Callable[[models.IndexView], None]


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def ledger(request: pytest.FixtureRequest) -> Iterator[tuple[IndexLedgerReader, Seed]]:
    if request.param == "fake":
        fake = FakeIndexLedger()
        yield fake, fake.seed
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    collection = client[database][INDEXES_COLLECTION]

    def seed(index: models.IndexView) -> None:
        collection.insert_one(index_document(index))

    yield MongoIndexLedger(client, database), seed
    client.drop_database(database)


OLD = make_index_view("aaaaaaaaaaaa", created_at=T0 - timedelta(days=1))
NEW = make_index_view("bbbbbbbbbbbb", created_at=T0, chunking_fingerprint="c7a1e2f3b4d5")


class TestGivenEmptyLedger:
    """No index prepared yet."""

    def test_when_asked_then_nothing(self, ledger: tuple[IndexLedgerReader, Seed]) -> None:
        reader, _ = ledger

        assert reader.get_index("aaaaaaaaaaaa") is None
        assert reader.newest_index() is None


class TestGivenTwoIndexes:
    """Two fingerprints, prepared a day apart."""

    @pytest.fixture
    def seeded(self, ledger: tuple[IndexLedgerReader, Seed]) -> IndexLedgerReader:
        reader, seed = ledger
        seed(OLD)
        seed(NEW)
        return reader

    def test_when_fingerprint_asked_then_that_index(self, seeded: IndexLedgerReader) -> None:
        assert seeded.get_index("aaaaaaaaaaaa") == OLD
        assert seeded.get_index("bbbbbbbbbbbb") == NEW
        assert seeded.get_index("cccccccccccc") is None

    def test_when_newest_asked_then_latest_created(self, seeded: IndexLedgerReader) -> None:
        newest = seeded.newest_index()

        assert newest == NEW
        assert newest.created_at.tzinfo is not None


class TestGivenMalformedLedger:
    """A collection answering records this role cannot decode, or failing."""

    @pytest.mark.parametrize(
        "document, message",
        [
            ({**index_document(OLD), "schema_version": 2}, "schema_version 2"),
            ({"_id": "aaaaaaaaaaaa", "schema_version": 1}, "cannot be decoded"),
        ],
    )
    def test_when_record_unusable_then_index_unavailable(
        self, document: Document, message: str
    ) -> None:
        reader = MongoIndexLedger(_stub(document), "db")

        with pytest.raises(IndexUnavailableError, match=message):
            reader.get_index("aaaaaaaaaaaa")

    def test_when_server_fails_then_index_unavailable(self) -> None:
        reader = MongoIndexLedger(_stub(PyMongoError("gone")), "db")

        with pytest.raises(IndexUnavailableError, match="cannot be read"):
            reader.get_index("aaaaaaaaaaaa")
        with pytest.raises(IndexUnavailableError, match="cannot be read"):
            reader.newest_index()

    def test_when_schema_version_differs_then_decoder_rejects(self) -> None:
        with pytest.raises(ValueError, match="schema_version 2"):
            index_from_document({**index_document(OLD), "schema_version": 2})


def _stub(answer: Document | Exception) -> MongoClient[Document]:
    """A stub in the client's place: the ledger only subscripts it."""
    return cast("MongoClient[Document]", _StubClient(answer))


class _StubCollection:
    def __init__(self, answer: Document | Exception) -> None:
        self._answer = answer

    def find_one(self, *args: Any, **kwargs: Any) -> Document:  # noqa: ANN401 - pymongo's signature
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


class _StubClient:
    """Enough of `MongoClient` for the ledger: ``client[database][collection]``."""

    def __init__(self, answer: Document | Exception) -> None:
        self._collection = _StubCollection(answer)

    def __getitem__(self, database: str) -> dict[str, _StubCollection]:
        return {INDEXES_COLLECTION: self._collection}
