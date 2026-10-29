"""Semantics of the Search Index, on the fake and on Qdrant (`integration`, `qdrant`)."""

import dataclasses
import uuid
from collections.abc import Iterator

import pytest

from scene_rag.retrieval.embedding.protocols import IndexMismatchError, SearchIndex
from scene_rag.retrieval.embedding.qdrant_index import QdrantSearchIndex
from tests.support.embedding_fakes import (
    FakeSearchIndex,
    make_chunk_views,
    make_point,
    sha256_version,
)

FIRST = make_chunk_views(sha256_version(b"first"))
SECOND = make_chunk_views(sha256_version(b"second"), count=2)


@pytest.fixture(
    params=["fake", pytest.param("qdrant", marks=[pytest.mark.integration, pytest.mark.qdrant])]
)
def index(request: pytest.FixtureRequest) -> Iterator[SearchIndex]:
    if request.param == "fake":
        yield FakeSearchIndex()
        return
    qdrant: QdrantSearchIndex = request.getfixturevalue("qdrant_index")
    yield qdrant


@pytest.fixture
def collection(index: SearchIndex) -> Iterator[str]:
    name = f"test_{uuid.uuid4().hex[:8]}"
    yield name
    index.drop_collection(name)  # type: ignore[attr-defined]  # both backends offer it for tests


class TestGivenNoCollection:
    """Nothing created yet."""

    def test_when_described_then_none(self, index: SearchIndex, collection: str) -> None:
        assert index.describe(collection) is None

    def test_when_ensured_then_described_empty(self, index: SearchIndex, collection: str) -> None:
        index.ensure_collection(collection, 8, "cosine")

        info = index.describe(collection)
        assert info is not None
        assert (info.dimensions, info.distance, info.point_count) == (8, "cosine", 0)

    def test_when_ensured_twice_alike_then_fine(self, index: SearchIndex, collection: str) -> None:
        index.ensure_collection(collection, 8, "dot")
        index.ensure_collection(collection, 8, "dot")

    @pytest.mark.parametrize("dimensions, distance", [(4, "cosine"), (8, "euclid")])
    def test_when_ensured_differently_then_mismatch(
        self, index: SearchIndex, collection: str, dimensions: int, distance: str
    ) -> None:
        index.ensure_collection(collection, 8, "cosine")

        with pytest.raises(IndexMismatchError, match=collection):
            index.ensure_collection(collection, dimensions, distance)


class TestGivenCollection:
    """A collection with the points of two chunk sets."""

    @pytest.fixture
    def filled(self, index: SearchIndex, collection: str) -> None:
        index.ensure_collection(collection, 8, "cosine")
        index.upsert(collection, [make_point(c) for c in FIRST])
        index.upsert(collection, [make_point(c) for c in SECOND])

    def test_when_counted_then_all_or_per_chunk_set(
        self, index: SearchIndex, collection: str, filled: None
    ) -> None:
        assert index.count(collection) == 5
        first_set = make_point(FIRST[0]).entry.chunk_set_version
        assert index.count(collection, chunk_set_version=first_set) == 3
        assert index.count(collection, chunk_set_version="sha256:00") == 0
        info = index.describe(collection)
        assert info is not None and info.point_count == 5

    def test_when_entries_fetched_then_payload_round_trips_in_asked_order(
        self, index: SearchIndex, collection: str, filled: None
    ) -> None:
        points = [make_point(c) for c in FIRST]

        entries = index.get_entries(
            collection, [points[2].id, "00000000-0000-5000-8000-000000000000", points[0].id]
        )

        assert entries == [points[2].entry, points[0].entry]

    def test_when_upserted_again_then_replaced_not_duplicated(
        self, index: SearchIndex, collection: str, filled: None
    ) -> None:
        point = make_point(FIRST[1])
        changed = dataclasses.replace(point, entry=dataclasses.replace(point.entry, text="other"))

        index.upsert(collection, [changed])

        assert index.count(collection) == 5
        assert index.get_entries(collection, [point.id]) == [changed.entry]
