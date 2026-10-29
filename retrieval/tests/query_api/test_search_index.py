"""The Qdrant reader: request shapes over a fake transport, semantics on the fake and on Qdrant.

The Qdrant cases (`integration`, `qdrant`) are seeded through the embedding step's writer.
"""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from scene_rag.retrieval.embedding import models as embedding_models
from scene_rag.retrieval.embedding.qdrant_index import QdrantSearchIndex, entry_to_payload
from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.http_transport import HttpStatusError, UrllibTransport
from scene_rag.retrieval.query_api.protocols import IndexUnavailableError, SearchIndexReader
from scene_rag.retrieval.query_api.search_index import (
    QdrantSearchReader,
    entry_from_payload,
    search_filter,
)
from tests.support import embedding_fakes
from tests.support.query_fakes import (
    PUBLIC,
    TEAM,
    FakeSearchReader,
    FakeTransport,
    fake_vector,
    make_entry_view,
    make_payload,
)

BASE = "http://qdrant:6333"
COLLECTION = "chunks_0123456789ab"
ENTRY = make_entry_view()


def collection_answer(size: int = 8, distance: str = "Cosine", points: int = 3) -> dict[str, Any]:
    return {
        "result": {
            "status": "green",
            "points_count": points,
            "config": {"params": {"vectors": {"size": size, "distance": distance}}},
        }
    }


def query_answer(payload: dict[str, Any] | None) -> dict[str, Any]:
    assert payload is not None
    return {
        "result": {
            "points": [
                {"id": "p1", "score": 0.91, "payload": make_payload(ENTRY)},
                {"id": "p2", "score": 0.5, "payload": make_payload(make_entry_view(1))},
            ]
        }
    }


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport(
        {
            "GET /": {"title": "qdrant - vector search engine", "version": "1.20.0"},
            f"GET /collections/{COLLECTION}": collection_answer(),
            f"POST /collections/{COLLECTION}/points/query": query_answer,
        },
        base=BASE,
    )


@pytest.fixture
def reader(transport: FakeTransport) -> QdrantSearchReader:
    return QdrantSearchReader(BASE + "/", transport, timeout_seconds=42.0)


class TestGivenServer:
    """A reachable server with one collection."""

    def test_when_version_asked_then_root_probed_with_short_timeout(
        self, reader: QdrantSearchReader, transport: FakeTransport
    ) -> None:
        assert reader.version() == "1.20.0"
        assert transport.calls == [("GET", BASE + "/", None, 5.0)]

    def test_when_collection_described_then_in_the_steps_names(
        self, reader: QdrantSearchReader
    ) -> None:
        assert reader.describe(COLLECTION) == models.CollectionInfo(8, "cosine", 3)

    def test_when_collection_absent_then_none(self, reader: QdrantSearchReader) -> None:
        assert reader.describe("chunks_ffffffffffff") is None

    def test_when_searched_then_query_request_filters_before_ranking(
        self, reader: QdrantSearchReader, transport: FakeTransport
    ) -> None:
        vector = fake_vector("q")

        hits = reader.search(
            COLLECTION,
            vector,
            limit=5,
            entitlements=frozenset({"team-platform", "public"}),
            asset_version="sha256:00",
        )

        assert [(h.entry.chunk_version, h.score) for h in hits] == [
            (ENTRY.chunk_version, 0.91),
            (make_entry_view(1).chunk_version, 0.5),
        ]
        assert hits[0].entry == ENTRY
        (method, url, payload, timeout) = transport.calls[-1]
        assert (method, url, timeout) == (
            "POST",
            f"{BASE}/collections/{COLLECTION}/points/query",
            42.0,
        )
        assert payload == {
            "query": list(vector),
            "limit": 5,
            "filter": {
                "must": [
                    {"key": "entitlements", "match": {"any": ["public", "team-platform"]}},
                    {"key": "asset_version", "match": {"value": "sha256:00"}},
                ]
            },
            "with_payload": True,
            "with_vector": False,
        }

    def test_when_no_asset_filter_then_entitlements_only(self) -> None:
        assert search_filter(PUBLIC, None) == {
            "must": [{"key": "entitlements", "match": {"any": ["public"]}}]
        }

    def test_when_no_entitlements_then_nothing_asked(
        self, reader: QdrantSearchReader, transport: FakeTransport
    ) -> None:
        hits = reader.search(
            COLLECTION, fake_vector("q"), limit=5, entitlements=frozenset(), asset_version=None
        )

        assert hits == []
        assert transport.calls == []

    @pytest.mark.parametrize(
        "answer",
        [{"result": {}}, {"result": {"points": [{"id": "p", "score": "x"}]}}, {"status": "ok"}],
    )
    def test_when_answer_unusable_then_index_unavailable(
        self, reader: QdrantSearchReader, transport: FakeTransport, answer: dict[str, Any]
    ) -> None:
        transport.answers[f"POST /collections/{COLLECTION}/points/query"] = answer

        with pytest.raises(IndexUnavailableError, match="unusable"):
            reader.search(COLLECTION, (1.0,), limit=1, entitlements=PUBLIC, asset_version=None)

    def test_when_payload_schema_differs_then_index_unavailable(
        self, reader: QdrantSearchReader, transport: FakeTransport
    ) -> None:
        payload = {**make_payload(ENTRY), "schema_version": 2}
        transport.answers[f"POST /collections/{COLLECTION}/points/query"] = {
            "result": {"points": [{"id": "p", "score": 0.1, "payload": payload}]}
        }

        with pytest.raises(IndexUnavailableError, match="schema_version 2"):
            reader.search(COLLECTION, (1.0,), limit=1, entitlements=PUBLIC, asset_version=None)

    def test_when_description_unusable_then_index_unavailable(
        self, reader: QdrantSearchReader, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /collections/{COLLECTION}"] = {"result": {"config": {}}}

        with pytest.raises(IndexUnavailableError, match="unusable"):
            reader.describe(COLLECTION)


class TestGivenFailingServer:
    """Error statuses and a server that is down."""

    def test_when_search_answers_error_status_then_index_unavailable(
        self, reader: QdrantSearchReader
    ) -> None:
        with pytest.raises(IndexUnavailableError, match="HTTP 404"):
            reader.search(
                "chunks_ffffffffffff", (1.0,), limit=1, entitlements=PUBLIC, asset_version=None
            )

    def test_when_description_answers_other_error_then_index_unavailable(
        self, reader: QdrantSearchReader, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /collections/{COLLECTION}"] = HttpStatusError(500, BASE, "boom")

        with pytest.raises(IndexUnavailableError, match="HTTP 500"):
            reader.describe(COLLECTION)

    def test_when_unreachable_then_index_unavailable(self, transport: FakeTransport) -> None:
        transport.unreachable("GET /")

        with pytest.raises(IndexUnavailableError, match="could not reach"):
            QdrantSearchReader(BASE, transport, 1.0).version()


# --- semantics, fake and real ---------------------------------------------------------------

FIRST = embedding_fakes.make_chunk_views(embedding_fakes.sha256_version(b"first"))
SECOND = embedding_fakes.make_chunk_views(
    embedding_fakes.sha256_version(b"second"), count=2, entitlements=TEAM
)
POINTS = [embedding_fakes.make_point(c) for c in (*FIRST, *SECOND)]


def view_of(point: embedding_models.Point) -> models.IndexEntryView:
    """The query role's view of a point the embedding step wrote: decoder over encoder."""
    return entry_from_payload(entry_to_payload(point.entry))


@pytest.fixture(
    params=["fake", pytest.param("qdrant", marks=[pytest.mark.integration, pytest.mark.qdrant])]
)
def filled(request: pytest.FixtureRequest) -> Iterator[tuple[SearchIndexReader, str]]:
    collection = f"test_{uuid.uuid4().hex[:8]}"
    if request.param == "fake":
        fake = FakeSearchReader()
        fake.seed(collection, [(p.vector, view_of(p)) for p in POINTS])
        yield fake, collection
        return
    writer: QdrantSearchIndex = request.getfixturevalue("qdrant_index")
    writer.ensure_collection(collection, embedding_fakes.DIMENSIONS, "cosine")
    writer.upsert(collection, POINTS)
    url: str = request.getfixturevalue("qdrant_url")
    yield QdrantSearchReader(url, UrllibTransport(), timeout_seconds=5.0), collection
    writer.drop_collection(collection)


class TestGivenFilledCollection:
    """Three public points of one asset, two team points of another."""

    def test_when_searched_with_a_points_text_then_that_point_first(
        self, filled: tuple[SearchIndexReader, str]
    ) -> None:
        reader, collection = filled

        hits = reader.search(
            collection, POINTS[1].vector, limit=3, entitlements=PUBLIC, asset_version=None
        )

        assert [h.entry.chunk_version for h in hits][0] == FIRST[1].version
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)
        assert hits[0].entry == view_of(POINTS[1])
        assert len(hits) == 3
        assert all(h.entry.entitlements == PUBLIC for h in hits)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_when_caller_shares_an_entitlement_then_those_points_only(
        self, filled: tuple[SearchIndexReader, str]
    ) -> None:
        reader, collection = filled

        hits = reader.search(
            collection,
            POINTS[0].vector,
            limit=10,
            entitlements=frozenset({"team-platform", "other"}),
            asset_version=None,
        )

        assert {h.entry.chunk_version for h in hits} == {c.version for c in SECOND}

    def test_when_limited_then_at_most_that_many(
        self, filled: tuple[SearchIndexReader, str]
    ) -> None:
        reader, collection = filled

        hits = reader.search(
            collection, POINTS[0].vector, limit=2, entitlements=PUBLIC, asset_version=None
        )

        assert len(hits) == 2

    def test_when_asset_version_given_then_that_asset_only(
        self, filled: tuple[SearchIndexReader, str]
    ) -> None:
        reader, collection = filled

        hits = reader.search(
            collection,
            POINTS[0].vector,
            limit=10,
            entitlements=frozenset({"public", "team-platform"}),
            asset_version=SECOND[0].asset_version,
        )

        assert {h.entry.chunk_version for h in hits} == {c.version for c in SECOND}

    def test_when_no_entitlements_then_nothing(self, filled: tuple[SearchIndexReader, str]) -> None:
        reader, collection = filled

        assert (
            reader.search(
                collection, POINTS[0].vector, limit=10, entitlements=frozenset(), asset_version=None
            )
            == []
        )

    def test_when_described_then_dimensions_and_count(
        self, filled: tuple[SearchIndexReader, str]
    ) -> None:
        reader, collection = filled

        info = reader.describe(collection)

        assert info is not None
        assert (info.dimensions, info.distance, info.point_count) == (8, "cosine", 5)
        assert reader.describe("chunks_000000000000") is None
