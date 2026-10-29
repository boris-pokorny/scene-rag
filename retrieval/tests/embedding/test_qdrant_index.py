"""The Qdrant backend over a fake transport: request shapes and answer decoding."""

from typing import Any

import pytest

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.http_transport import HttpStatusError
from scene_rag.retrieval.embedding.protocols import IndexMismatchError, SearchIndexError
from scene_rag.retrieval.embedding.qdrant_index import (
    PAYLOAD_INDEXES,
    QdrantSearchIndex,
    entry_to_payload,
)
from tests.support.embedding_fakes import FakeTransport, make_chunk_views, make_point

BASE = "http://qdrant:6333"
COLLECTION = "chunks_0123456789ab"
CHUNKS = make_chunk_views()


def collection_answer(size: int = 8, distance: str = "Cosine", points: int = 3) -> dict[str, Any]:
    return {
        "result": {
            "status": "green",
            "points_count": points,
            "config": {"params": {"vectors": {"size": size, "distance": distance}}},
        },
        "status": "ok",
    }


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport(
        {
            "GET /": {"title": "qdrant - vector search engine", "version": "1.15.4"},
            f"PUT /collections/{COLLECTION}": {"result": True},
            f"PUT /collections/{COLLECTION}/index": {"result": {"status": "acknowledged"}},
            f"PUT /collections/{COLLECTION}/points?wait=true": {"result": {"status": "completed"}},
            f"POST /collections/{COLLECTION}/points/count": {"result": {"count": 2}},
            f"DELETE /collections/{COLLECTION}": {"result": True},
        },
        base=BASE,
    )


@pytest.fixture
def index(transport: FakeTransport) -> QdrantSearchIndex:
    return QdrantSearchIndex(BASE + "/", transport, timeout_seconds=42.0)


class TestGivenServer:
    """A reachable server."""

    def test_when_version_asked_then_root_probed_with_short_timeout(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        assert index.version() == "1.15.4"
        assert transport.calls == [("GET", BASE + "/", None, 5.0)]

    def test_when_collection_absent_then_describe_none(self, index: QdrantSearchIndex) -> None:
        assert index.describe(COLLECTION) is None

    def test_when_collection_present_then_described_in_the_steps_names(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /collections/{COLLECTION}"] = collection_answer(768, "Dot", 5)

        assert index.describe(COLLECTION) == models.CollectionInfo(768, "dot", 5)

    def test_when_collection_absent_then_created_with_payload_indexes(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        index.ensure_collection(COLLECTION, 8, "cosine")

        requests = [(m, u.removeprefix(BASE), p) for m, u, p, _ in transport.calls]
        assert requests[0] == ("GET", f"/collections/{COLLECTION}", None)
        assert requests[1] == (
            "PUT",
            f"/collections/{COLLECTION}",
            {"vectors": {"size": 8, "distance": "Cosine"}},
        )
        assert requests[2:] == [
            (
                "PUT",
                f"/collections/{COLLECTION}/index",
                {"field_name": f, "field_schema": "keyword"},
            )
            for f in PAYLOAD_INDEXES
        ]

    def test_when_collection_matches_then_verified_not_created(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /collections/{COLLECTION}"] = collection_answer(8, "Cosine")

        index.ensure_collection(COLLECTION, 8, "cosine")

        assert not any(
            m == "PUT" and u == f"{BASE}/collections/{COLLECTION}" for m, u, _, _ in transport.calls
        )

    @pytest.mark.parametrize("size, distance", [(4, "Cosine"), (8, "Euclid")])
    def test_when_collection_differs_then_mismatch(
        self, index: QdrantSearchIndex, transport: FakeTransport, size: int, distance: str
    ) -> None:
        transport.answers[f"GET /collections/{COLLECTION}"] = collection_answer(size, distance)

        with pytest.raises(IndexMismatchError, match=COLLECTION):
            index.ensure_collection(COLLECTION, 8, "cosine")

    def test_when_created_by_another_process_meanwhile_then_verified(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        def race(payload: dict[str, Any] | None) -> dict[str, Any]:
            transport.answers[f"GET /collections/{COLLECTION}"] = collection_answer(8, "Cosine")
            raise HttpStatusError(409, f"{BASE}/collections/{COLLECTION}", "already exists")

        transport.answers[f"PUT /collections/{COLLECTION}"] = race

        index.ensure_collection(COLLECTION, 8, "cosine")

    def test_when_payload_index_refused_then_warned_not_failed(
        self, index: QdrantSearchIndex, transport: FakeTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        transport.answers[f"PUT /collections/{COLLECTION}/index"] = HttpStatusError(
            400, f"{BASE}/collections/{COLLECTION}/index", "already exists"
        )

        index.ensure_collection(COLLECTION, 8, "cosine")

        assert "payload index on entitlements" in caplog.text

    def test_when_creation_refused_then_error(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        transport.answers[f"PUT /collections/{COLLECTION}"] = HttpStatusError(
            400, f"{BASE}/collections/{COLLECTION}", "bad request"
        )

        with pytest.raises(SearchIndexError, match="HTTP 400"):
            index.ensure_collection(COLLECTION, 8, "cosine")

    def test_when_upserted_then_points_sent_with_wait(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        points = [make_point(c) for c in CHUNKS[:2]]

        index.upsert(COLLECTION, points)
        index.upsert(COLLECTION, [])

        ((method, url, payload, timeout),) = transport.calls
        assert (method, url, timeout) == (
            "PUT",
            f"{BASE}/collections/{COLLECTION}/points?wait=true",
            42.0,
        )
        assert payload == {
            "points": [
                {"id": p.id, "vector": list(p.vector), "payload": entry_to_payload(p.entry)}
                for p in points
            ]
        }

    def test_when_counted_then_exact_with_optional_filter(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        assert index.count(COLLECTION) == 2
        assert index.count(COLLECTION, chunk_set_version="sha256:ab") == 2

        payloads = [p for _, _, p, _ in transport.calls]
        assert payloads[0] == {"exact": True}
        assert payloads[1] == {
            "exact": True,
            "filter": {"must": [{"key": "chunk_set_version", "match": {"value": "sha256:ab"}}]},
        }

    def test_when_entries_fetched_then_in_asked_order_skipping_absent(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        points = [make_point(c) for c in CHUNKS]
        transport.answers[f"POST /collections/{COLLECTION}/points"] = {
            "result": [{"id": p.id, "payload": entry_to_payload(p.entry)} for p in points[:2]]
        }

        entries = index.get_entries(COLLECTION, [points[1].id, points[2].id, points[0].id])

        assert entries == [points[1].entry, points[0].entry]
        assert index.get_entries(COLLECTION, []) == []
        (_, _, payload, _) = transport.calls[0]
        assert payload == {
            "ids": [points[1].id, points[2].id, points[0].id],
            "with_payload": True,
            "with_vector": False,
        }

    def test_when_dropped_then_delete_sent_and_absent_is_fine(
        self, index: QdrantSearchIndex, transport: FakeTransport
    ) -> None:
        index.drop_collection(COLLECTION)
        index.drop_collection("chunks_000000000000")

        assert [m for m, _, _, _ in transport.calls] == ["DELETE", "DELETE"]


class TestGivenUnreachableServer:
    """A transport that fails."""

    def test_when_version_asked_then_search_index_error(self, transport: FakeTransport) -> None:
        transport.unreachable("GET /")

        with pytest.raises(SearchIndexError, match="could not reach"):
            QdrantSearchIndex(BASE, transport, 1.0).version()

    def test_when_describe_fails_otherwise_then_search_index_error(
        self, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /collections/{COLLECTION}"] = HttpStatusError(
            500, f"{BASE}/collections/{COLLECTION}", "broken"
        )

        with pytest.raises(SearchIndexError, match="HTTP 500"):
            QdrantSearchIndex(BASE, transport, 1.0).describe(COLLECTION)

    def test_when_drop_fails_otherwise_then_search_index_error(
        self, transport: FakeTransport
    ) -> None:
        transport.answers[f"DELETE /collections/{COLLECTION}"] = HttpStatusError(
            500, f"{BASE}/collections/{COLLECTION}", "broken"
        )

        with pytest.raises(SearchIndexError, match="HTTP 500"):
            QdrantSearchIndex(BASE, transport, 1.0).drop_collection(COLLECTION)

    def test_when_version_status_error_then_search_index_error(
        self, transport: FakeTransport
    ) -> None:
        transport.answers["GET /"] = HttpStatusError(503, BASE + "/", "starting")

        with pytest.raises(SearchIndexError, match="HTTP 503"):
            QdrantSearchIndex(BASE, transport, 1.0).version()


class TestGivenUnusableAnswers:
    """A server that answers a status of success with a body of another shape."""

    def test_when_collection_has_named_vectors_then_search_index_error(
        self, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /collections/{COLLECTION}"] = {
            "result": {
                "points_count": 0,
                "config": {"params": {"vectors": {"text": {"size": 8, "distance": "Cosine"}}}},
            }
        }

        with pytest.raises(SearchIndexError, match=f"/collections/{COLLECTION} answered"):
            QdrantSearchIndex(BASE, transport, 1.0).describe(COLLECTION)

    def test_when_count_answer_has_no_result_then_search_index_error(
        self, transport: FakeTransport
    ) -> None:
        transport.answers[f"POST /collections/{COLLECTION}/points/count"] = {"status": "ok"}

        with pytest.raises(SearchIndexError, match="points/count answered"):
            QdrantSearchIndex(BASE, transport, 1.0).count(COLLECTION)

    def test_when_points_answer_is_not_a_list_then_search_index_error(
        self, transport: FakeTransport
    ) -> None:
        transport.answers[f"POST /collections/{COLLECTION}/points"] = {"result": {"id": "x"}}

        with pytest.raises(SearchIndexError, match="points answered"):
            QdrantSearchIndex(BASE, transport, 1.0).get_entries(COLLECTION, ["x"])
