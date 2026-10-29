"""Fakes and builders for the tests of the query-api role."""

import dataclasses
import hashlib
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.config import QueryConfig
from scene_rag.retrieval.query_api.http_transport import HttpStatusError, Json, TransportError
from scene_rag.retrieval.query_api.prompt import PromptBuilder
from scene_rag.retrieval.query_api.protocols import (
    AnswerGenerator,
    IndexUnavailableError,
    ModelError,
    NotFoundError,
    RefusalPolicy,
    Reranker,
    RetiredFilter,
    Retriever,
)
from scene_rag.retrieval.query_api.refusal import ScoreThresholdPolicy
from scene_rag.retrieval.query_api.reranker import NoReranker
from scene_rag.retrieval.query_api.retriever import DenseRetriever
from scene_rag.retrieval.query_api.service import QueryServiceImpl

T0 = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
EMBEDDING_FP = "5e2d4c6b8a90"
CHUNKING_FP = "c7a1e2f3b4d5"
QUERY_CONFIG = "11d0a9e2c4f1"
DIMENSIONS = 8
EMBEDDING_MODEL = models.ModelRef("embeddinggemma", "sha256:" + "ab" * 32)
GENERATOR_MODEL = models.ModelRef("qwen2.5:3b", "sha256:" + "cd" * 32)
QUERY_PREFIX = "task: search result | query: "
PUBLIC = frozenset({"public"})
TEAM = frozenset({"team-platform"})
USAGE = models.Usage(100, 12)
# The host the test client sends; the application serves it alone.
TEST_HOSTS = ("testserver",)
SOURCES = (models.SourceRef("local", "asset-0000000000000001", "clip.mp4"),)


def sha256_version(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


ASSET = sha256_version(b"abcd")
OTHER_ASSET = sha256_version(b"efgh")


# --- builders ------------------------------------------------------------------------------


def make_index_view(
    fingerprint: str = EMBEDDING_FP,
    *,
    dimensions: int = DIMENSIONS,
    model: models.ModelRef = EMBEDDING_MODEL,
    chunking_fingerprint: str | None = None,
    created_at: datetime = T0,
) -> models.IndexView:
    return models.IndexView(
        fingerprint=fingerprint,
        collection="chunks_" + fingerprint,
        model=model,
        dimensions=dimensions,
        distance="cosine",
        chunking_fingerprint=chunking_fingerprint,
        query_prefix=QUERY_PREFIX,
        truncate=True,
        num_ctx=2048,
        created_at=created_at,
    )


def index_document(index: models.IndexView) -> Json:
    """The ledger document of a view, as the embedding step writes it."""
    return {
        "_id": index.fingerprint,
        "schema_version": models.INDEX_SCHEMA_VERSION,
        "collection": index.collection,
        "model": {"name": index.model.name, "digest": index.model.digest},
        "dimensions": index.dimensions,
        "distance": index.distance,
        "chunking_fingerprint": index.chunking_fingerprint,
        "document_prefix": "title: none | text: ",
        "query_prefix": index.query_prefix,
        "truncate": index.truncate,
        "num_ctx": index.num_ctx,
        "created_at": index.created_at,
        "image": None,
    }


def make_entry_view(
    index: int = 0,
    *,
    asset_version: str = ASSET,
    text: str | None = None,
    entitlements: frozenset[str] = PUBLIC,
    chunking_fingerprint: str = CHUNKING_FP,
) -> models.IndexEntryView:
    scene_set = sha256_version(f"{asset_version}\nscene-set".encode())
    return models.IndexEntryView(
        chunk_version=sha256_version(f"{scene_set}\n{chunking_fingerprint}\n{index}".encode()),
        scene_set_version=scene_set,
        asset_version=asset_version,
        chunking_fingerprint=chunking_fingerprint,
        scene_indices=(index,),
        start_seconds=index * 4.0,
        end_seconds=index * 4.0 + 4.0,
        text=f"chunk {index} text" if text is None else text,
        sources=SOURCES,
        entitlements=entitlements,
    )


def make_payload(entry: models.IndexEntryView) -> Json:
    """The point payload of a view, as the embedding step writes it."""
    return {
        "version": sha256_version(entry.chunk_version.encode()),
        "schema_version": models.INDEX_ENTRY_SCHEMA_VERSION,
        "chunk_version": entry.chunk_version,
        "chunk_set_version": sha256_version(entry.scene_set_version.encode()),
        "scene_set_version": entry.scene_set_version,
        "asset_version": entry.asset_version,
        "chunking_fingerprint": entry.chunking_fingerprint,
        "segmentation_fingerprint": "3b9e1f6a2c40",
        "chunk_index": entry.scene_indices[0],
        "scene_indices": list(entry.scene_indices),
        "start_seconds": entry.start_seconds,
        "end_seconds": entry.end_seconds,
        "text": entry.text,
        "sources": [
            {"source_id": s.source_id, "asset_id": s.asset_id, "filename": s.filename}
            for s in entry.sources
        ],
        "entitlements": sorted(entry.entitlements),
        "classifications": ["internal"],
        "created_at": "2026-10-28T12:00:00Z",
        "lineage": {
            "step": "embedding",
            "fingerprint": EMBEDDING_FP,
            "inputs": ["chunk:" + entry.chunk_version],
            "version": sha256_version(entry.chunk_version.encode()),
            "image": None,
        },
    }


def make_source(entry: models.IndexEntryView, score: float) -> models.Source:
    return models.Source(
        chunk_version=entry.chunk_version,
        asset_version=entry.asset_version,
        scene_set_version=entry.scene_set_version,
        scene_indices=entry.scene_indices,
        start_seconds=entry.start_seconds,
        end_seconds=entry.end_seconds,
        score=score,
        text=entry.text,
        sources=entry.sources,
        entitlements=entry.entitlements,
    )


def make_served(
    query_config: str = QUERY_CONFIG,
    embedding_fingerprint: str = EMBEDDING_FP,
    chunking_fingerprint: str | None = None,
) -> models.Served:
    return models.Served(
        query_config=query_config,
        embedding_fingerprint=embedding_fingerprint,
        chunking_fingerprint=chunking_fingerprint,
        collection="chunks_" + embedding_fingerprint,
    )


SERVED = make_served()


def fake_vector(text: str, dimensions: int = DIMENSIONS) -> models.Vector:
    """A unit vector determined by the text alone: equal texts embed equally."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] / 127.5 - 1.0 for i in range(dimensions)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return tuple(round(v / norm, 6) for v in raw)


def cosine(a: models.Vector, b: models.Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


# --- fakes ---------------------------------------------------------------------------------

type Answer = Json | Exception | Callable[[Json | None], Json]
type Body = tuple[bytes, str] | Exception


class FakeTransport:
    """Transport answering from scripts keyed by ``METHOD path``, recording every call.

    A JSON answer is a document, an exception to raise, or a function of the payload; a body
    answer (``FETCH path``) is ``(bytes, media type)`` or an exception. A path without a
    script entry answers 404. With a ``base``, the key's path is the URL without it; without
    one, it is the URL's path and query, whatever the host, so one script serves several
    backends.
    """

    def __init__(
        self,
        answers: Mapping[str, Answer] | None = None,
        bodies: Mapping[str, Body] | None = None,
        base: str = "",
    ) -> None:
        self.answers: dict[str, Answer] = dict(answers or {})
        self.bodies: dict[str, Body] = dict(bodies or {})
        self.base = base
        self.calls: list[tuple[str, str, Json | None, float]] = []
        self._lock = threading.Lock()

    def _path(self, url: str) -> str:
        if self.base:
            return url.removeprefix(self.base)
        parts = urlsplit(url)
        return parts.path + (f"?{parts.query}" if parts.query else "")

    def exchange(self, method: str, url: str, payload: Json | None, timeout: float) -> Json:
        with self._lock:
            self.calls.append((method, url, payload, timeout))
        answer = self.answers.get(f"{method} {self._path(url)}")
        if answer is None:
            raise HttpStatusError(404, url, "not found")
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(payload)
        return answer

    def fetch(self, url: str, timeout: float) -> tuple[bytes, str]:
        with self._lock:
            self.calls.append(("FETCH", url, None, timeout))
        body = self.bodies.get(f"FETCH {self._path(url)}")
        if body is None:
            raise HttpStatusError(404, url, "not found")
        if isinstance(body, Exception):
            raise body
        return body

    def unreachable(self, *paths: str) -> None:
        """Make the given ``METHOD path`` keys fail as if the server were down."""
        for path in paths:
            error = TransportError(f"could not reach {self.base}{path}")
            if path.startswith("FETCH "):
                self.bodies[path] = error
            else:
                self.answers[path] = error


class FakeIndexLedger:
    """IndexLedgerReader over a dictionary of index records."""

    def __init__(self, *indexes: models.IndexView) -> None:
        self.indexes: dict[str, models.IndexView] = {}
        self.failure: Exception | None = None
        for index in indexes:
            self.seed(index)

    def seed(self, index: models.IndexView) -> None:
        self.indexes[index.fingerprint] = index

    def _check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def get_index(self, fingerprint: str) -> models.IndexView | None:
        self._check()
        return self.indexes.get(fingerprint)

    def newest_index(self) -> models.IndexView | None:
        self._check()
        if not self.indexes:
            return None
        return max(self.indexes.values(), key=lambda i: (i.created_at, i.fingerprint))


class FakeSearchReader:
    """SearchIndexReader over seeded points, with the semantics of the Qdrant backend.

    `tests/query/test_search_index.py` runs the same cases against both, so the fake stays
    faithful: cosine similarity, entitlement ``any`` filter, asset version filter, limit.
    """

    def __init__(self) -> None:
        self.collections: dict[str, tuple[int, str]] = {}
        self._points: dict[str, list[tuple[models.Vector, models.IndexEntryView]]] = {}
        self.searches: list[tuple[str, int, frozenset[str], str | None]] = []
        self.failure: Exception | None = None
        self._lock = threading.Lock()

    def seed(
        self,
        collection: str,
        points: Sequence[tuple[models.Vector, models.IndexEntryView]],
        *,
        dimensions: int = DIMENSIONS,
        distance: str = "cosine",
    ) -> None:
        with self._lock:
            self.collections.setdefault(collection, (dimensions, distance))
            self._points.setdefault(collection, []).extend(points)

    def _check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def version(self) -> str:
        self._check()
        return "fake"

    def describe(self, collection: str) -> models.CollectionInfo | None:
        self._check()
        if collection not in self.collections:
            return None
        dimensions, distance = self.collections[collection]
        return models.CollectionInfo(dimensions, distance, len(self._points[collection]))

    def search(
        self,
        collection: str,
        vector: models.Vector,
        *,
        limit: int,
        entitlements: frozenset[str],
        asset_version: str | None,
    ) -> Sequence[models.Hit]:
        self._check()
        with self._lock:
            self.searches.append((collection, limit, entitlements, asset_version))
        if collection not in self.collections:
            raise IndexUnavailableError(f"no collection {collection}")
        if not entitlements:
            return []
        hits = [
            models.Hit(entry=entry, score=round(cosine(vector, stored), 6))
            for stored, entry in self._points[collection]
            if entry.entitlements & entitlements and asset_version in (None, entry.asset_version)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


class FakeQueryEmbedder:
    """QueryEmbedder with vectors from the question's hash, or raising."""

    def __init__(
        self,
        index: models.IndexView | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self._index = make_index_view() if index is None else index
        self._fail = fail
        self.questions: list[str] = []
        self._lock = threading.Lock()

    @property
    def model(self) -> models.ModelRef:
        return self._index.model

    @property
    def dimensions(self) -> int:
        return self._index.dimensions

    def embed(self, question: str) -> models.Vector:
        with self._lock:
            self.questions.append(question)
        if self._fail:
            raise ModelError("embedding model down")
        return fake_vector(question, self._index.dimensions)


class FakeAnswerGenerator:
    """AnswerGenerator answering a scripted text, or raising; records every prompt."""

    def __init__(
        self,
        text: str = "The demo starts after the introduction [1].",
        *,
        usage: models.Usage = USAGE,
        failure: Exception | None = None,
        model: models.ModelRef = GENERATOR_MODEL,
    ) -> None:
        self.text = text
        self.usage = usage
        self.failure = failure
        self._model = model
        self.prompts: list[tuple[models.Message, ...]] = []
        self._lock = threading.Lock()

    @property
    def model(self) -> models.ModelRef:
        return self._model

    def generate(self, messages: Sequence[models.Message]) -> models.Generation:
        with self._lock:
            self.prompts.append(tuple(messages))
        if self.failure is not None:
            raise self.failure
        return models.Generation(self.text, self.usage)


class FakeSceneReader:
    """SceneReader over dictionaries of scene records and derivatives."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, int], Json] = {}
        self.derivatives: dict[str, models.DerivativeContent] = {}
        self.failure: Exception | None = None
        self.calls: list[tuple[str, ...]] = []

    def _check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def health(self) -> None:
        self._check()

    def get_scene_record(self, scene_set_version: str, index: int) -> Json | None:
        self.calls.append(("scene", scene_set_version, str(index)))
        self._check()
        return self.records.get((scene_set_version, index))

    def get_derivative(self, digest: str) -> models.DerivativeContent | None:
        self.calls.append(("derivative", digest))
        self._check()
        return self.derivatives.get(digest)


class FakeRetiredFilter:
    """RetiredFilter over a set of asset versions."""

    def __init__(self, *retired: str) -> None:
        self.retired = set(retired)

    def is_retired(self, asset_version: str) -> bool:
        return asset_version in self.retired


def as_document(value: Any) -> Any:  # noqa: ANN401 - walks untyped JSON
    """Render a store document the way the samples are written: dates as RFC 3339 strings."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: as_document(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_document(v) for v in value]
    return value


# --- harness -------------------------------------------------------------------------------


@dataclasses.dataclass
class QueryHarness:
    """The use cases wired to fakes, exposing them for assertions.

    `service` runs the real retriever, reranker, refusal policy and prompt builder over the
    fake embedder, Search Index, generator and scene-api. `variant` rebuilds it with some
    collaborators replaced.
    """

    embedder: FakeQueryEmbedder
    search_index: FakeSearchReader
    generator: FakeAnswerGenerator
    scenes: FakeSceneReader
    retired: FakeRetiredFilter
    config: QueryConfig
    index: models.IndexView
    served: models.Served
    service: QueryServiceImpl

    def seed(self, *entries: models.IndexEntryView) -> None:
        """Index the entries, each under the vector of its own text."""
        self.search_index.seed(
            self.index.collection,
            [(fake_vector(e.text, self.index.dimensions), e) for e in entries],
        )

    def variant(
        self,
        *,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        refusal: RefusalPolicy | None = None,
        generator: AnswerGenerator | None = None,
        retired: RetiredFilter | None = None,
        config: QueryConfig | None = None,
    ) -> QueryServiceImpl:
        config = self.config if config is None else config
        return QueryServiceImpl(
            retriever=retriever or DenseRetriever(self.embedder, self.search_index, self.index),
            reranker=reranker or NoReranker(),
            refusal=refusal or ScoreThresholdPolicy(config.refusal),
            prompt=PromptBuilder(config.prompt),
            generator=self.generator if generator is None else generator,
            scenes=self.scenes,
            retired=self.retired if retired is None else retired,
            config=config,
            served=self.served,
        )


def make_query_harness(
    *entries: models.IndexEntryView, config: QueryConfig | None = None
) -> QueryHarness:
    config = QueryConfig() if config is None else config
    index = make_index_view()
    embedder = FakeQueryEmbedder(index)
    search_index = FakeSearchReader()
    search_index.seed(index.collection, [])
    generator = FakeAnswerGenerator()
    scenes = FakeSceneReader()
    retired = FakeRetiredFilter()
    served = make_served(embedding_fingerprint=index.fingerprint)
    service = QueryServiceImpl(
        retriever=DenseRetriever(embedder, search_index, index),
        reranker=NoReranker(),
        refusal=ScoreThresholdPolicy(config.refusal),
        prompt=PromptBuilder(config.prompt),
        generator=generator,
        scenes=scenes,
        retired=retired,
        config=config,
        served=served,
    )
    harness = QueryHarness(
        embedder, search_index, generator, scenes, retired, config, index, served, service
    )
    harness.seed(*entries)
    return harness


class FakeQueryService:
    """QueryService answering scripted results, recording every call, or raising."""

    def __init__(
        self,
        *,
        search_result: models.SearchResult | None = None,
        answer_result: models.AnswerResult | None = None,
        records: Mapping[tuple[str, int], Json] | None = None,
        derivatives: Mapping[str, models.DerivativeContent] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.search_result = search_result or models.SearchResult((), SERVED)
        self.answer_result = answer_result or models.AnswerResult(
            "I cannot answer that from the indexed videos.",
            True,
            (),
            (),
            (),
            models.NO_USAGE,
            SERVED,
        )
        self.records = dict(records or {})
        self.derivatives = dict(derivatives or {})
        self.failure = failure
        self.calls: list[tuple[object, ...]] = []
        self._lock = threading.Lock()

    def _record(self, *call: object) -> None:
        with self._lock:
            self.calls.append(call)
        if self.failure is not None:
            raise self.failure

    @property
    def served(self) -> models.Served:
        return SERVED

    def search(self, query: models.SearchQuery) -> models.SearchResult:
        self._record("search", query)
        return self.search_result

    def answer(self, query: models.AnswerQuery) -> models.AnswerResult:
        self._record("answer", query)
        return self.answer_result

    def get_scene(self, scene_set_version: str, index: int, caller: models.Caller) -> Json:
        self._record("get_scene", scene_set_version, index, caller)
        record = self.records.get((scene_set_version, index))
        if record is None:
            raise NotFoundError("scene", f"{scene_set_version}/{index}")
        return record

    def get_keyframe(self, digest: str, caller: models.Caller) -> models.DerivativeContent:
        self._record("get_keyframe", digest, caller)
        content = self.derivatives.get(digest)
        if content is None:
            raise NotFoundError("keyframe", digest)
        return content
