"""Fakes and builders for the tests of the embedding step."""

import dataclasses
import hashlib
import math
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.config import EmbeddingConfig
from scene_rag.retrieval.embedding.dispatch import SerialDispatcher
from scene_rag.retrieval.embedding.http_transport import HttpStatusError, Json, TransportError
from scene_rag.retrieval.embedding.index_setup import prepare_index
from scene_rag.retrieval.embedding.protocols import (
    Dispatcher,
    EmbeddingModel,
    IndexMismatchError,
    IndexStoreWriter,
    ModelError,
    SearchIndexWriter,
)
from scene_rag.retrieval.embedding.report import Outcome
from scene_rag.retrieval.embedding.scheduler import Scheduler
from scene_rag.retrieval.embedding.service import EmbeddingService

T0 = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
SEGMENTATION_FP = "3b9e1f6a2c40"
CHUNKING_FP = "c7a1e2f3b4d5"
STEP_FP = "5e2d4c6b8a90"
DIMENSIONS = 8
MODEL = models.ModelRef("fake-embed", "sha256:" + "cd" * 32)
PUBLIC = frozenset({"public"})
SOURCES = (models.SourceView("local", "asset-0000000000000001", "clip.mp4"),)


def sha256_version(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class FakeClock:
    """Clock that returns a fixed time and advances only when told."""

    def __init__(self, start: datetime = T0) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current

    def tick(self, seconds: float = 60) -> datetime:
        self.current += timedelta(seconds=seconds)
        return self.current


# --- builders ------------------------------------------------------------------------------


def scene_set_version(asset_version: str, fingerprint: str = SEGMENTATION_FP) -> str:
    return sha256_version(f"{asset_version}\n{fingerprint}".encode())


def chunk_version(scene_set: str, index: int, fingerprint: str) -> str:
    return sha256_version(f"{scene_set}\n{fingerprint}\n{index}".encode())


def chunk_set_version(scene_set: str, fingerprint: str) -> str:
    return sha256_version(f"{scene_set}\n{fingerprint}".encode())


def make_chunk_views(
    asset_version: str = sha256_version(b"abcd"),
    *,
    count: int = 3,
    chunking_fingerprint: str = CHUNKING_FP,
    text: Callable[[int], str] = lambda i: f"chunk {i} text",
    entitlements: frozenset[str] = PUBLIC,
) -> tuple[models.ChunkView, ...]:
    """The chunks of one scene set under one chunking fingerprint, one scene each."""
    scene_set = scene_set_version(asset_version)
    return tuple(
        models.ChunkView(
            version=chunk_version(scene_set, index, chunking_fingerprint),
            scene_set_version=scene_set,
            asset_version=asset_version,
            segmentation_fingerprint=SEGMENTATION_FP,
            chunking_fingerprint=chunking_fingerprint,
            chunk_index=index,
            scene_indices=(index,),
            start_seconds=index * 4.0,
            end_seconds=index * 4.0 + 4.0,
            text=text(index),
            sources=SOURCES,
            entitlements=entitlements,
            classifications=("internal",),
        )
        for index in range(count)
    )


def make_chunk_set_view(
    chunks: Sequence[models.ChunkView] = (),
    *,
    asset_version: str = sha256_version(b"abcd"),
    chunking_fingerprint: str = CHUNKING_FP,
    chunk_count: int | None = None,
) -> models.ChunkSetView:
    """The chunk set of the chunks, or an empty one for the asset version."""
    if chunks:
        asset_version = chunks[0].asset_version
        chunking_fingerprint = chunks[0].chunking_fingerprint
    scene_set = scene_set_version(asset_version)
    return models.ChunkSetView(
        version=chunk_set_version(scene_set, chunking_fingerprint),
        scene_set_version=scene_set,
        asset_version=asset_version,
        chunking_fingerprint=chunking_fingerprint,
        chunk_count=len(chunks) if chunk_count is None else chunk_count,
        created_at=T0,
    )


def make_entry(
    chunk: models.ChunkView,
    chunk_set: models.ChunkSetView | None = None,
    fingerprint: str = STEP_FP,
    created_at: datetime = T0,
) -> models.IndexEntry:
    chunk_set = make_chunk_set_view([chunk]) if chunk_set is None else chunk_set
    version = models.index_entry_version(chunk.version, fingerprint, "sha256")
    return models.IndexEntry(
        version=version,
        chunk_version=chunk.version,
        chunk_set_version=chunk_set.version,
        scene_set_version=chunk.scene_set_version,
        asset_version=chunk.asset_version,
        chunking_fingerprint=chunk.chunking_fingerprint,
        segmentation_fingerprint=chunk.segmentation_fingerprint,
        chunk_index=chunk.chunk_index,
        scene_indices=chunk.scene_indices,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
        text=chunk.text,
        sources=chunk.sources,
        entitlements=chunk.entitlements,
        classifications=chunk.classifications,
        created_at=created_at,
        lineage=models.Lineage(
            fingerprint=fingerprint,
            inputs=(models.CHUNK_INPUT_PREFIX + chunk.version,),
            version=version,
        ),
    )


def make_point(chunk: models.ChunkView, fingerprint: str = STEP_FP) -> models.Point:
    return models.Point(
        id=models.point_id(chunk.version),
        vector=fake_vector(chunk.text),
        entry=make_entry(chunk, fingerprint=fingerprint),
    )


def make_index_set(
    chunks: Sequence[models.ChunkView] = (),
    *,
    chunk_set: models.ChunkSetView | None = None,
    fingerprint: str = STEP_FP,
    vector_count: int | None = None,
    created_at: datetime = T0,
) -> models.IndexSet:
    chunk_set = make_chunk_set_view(chunks) if chunk_set is None else chunk_set
    version = models.index_set_version(chunk_set.version, fingerprint, "sha256")
    return models.IndexSet(
        version=version,
        chunk_set_version=chunk_set.version,
        scene_set_version=chunk_set.scene_set_version,
        asset_version=chunk_set.asset_version,
        chunking_fingerprint=chunk_set.chunking_fingerprint,
        collection=models.collection_name(fingerprint),
        vector_count=len(chunks) if vector_count is None else vector_count,
        created_at=created_at,
        lineage=models.Lineage(
            fingerprint=fingerprint,
            inputs=tuple(models.CHUNK_INPUT_PREFIX + c.version for c in chunks),
            version=version,
        ),
    )


def make_index(
    fingerprint: str = STEP_FP,
    *,
    dimensions: int = DIMENSIONS,
    config: EmbeddingConfig | None = None,
    model: models.ModelRef = MODEL,
    created_at: datetime = T0,
) -> models.Index:
    config = EmbeddingConfig() if config is None else config
    return models.Index(
        fingerprint=fingerprint,
        collection=models.collection_name(fingerprint),
        model=model,
        dimensions=dimensions,
        distance=config.index.distance,
        chunking_fingerprint=config.upstream.chunking_fingerprint,
        document_prefix=config.model.document_prefix,
        query_prefix=config.model.query_prefix,
        truncate=config.model.truncate,
        num_ctx=config.model.num_ctx,
        created_at=created_at,
    )


def fake_vector(text: str, dimensions: int = DIMENSIONS) -> models.Vector:
    """A unit vector determined by the text alone: equal texts embed equally."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] / 127.5 - 1.0 for i in range(dimensions)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return tuple(round(v / norm, 6) for v in raw)


# --- fakes ---------------------------------------------------------------------------------


class FakeChunkReader:
    """ChunkReader over dictionaries of chunk sets and their chunks."""

    def __init__(self) -> None:
        self.chunk_sets: dict[str, models.ChunkSetView] = {}
        self.chunks: dict[str, tuple[models.ChunkView, ...]] = {}
        self.listings: list[str | None] = []
        self.fetches: list[tuple[str, str]] = []
        self.failure: Exception | None = None
        self._lock = threading.Lock()

    def seed(
        self, chunks: Sequence[models.ChunkView], chunk_set: models.ChunkSetView | None = None
    ) -> models.ChunkSetView:
        chunk_set = make_chunk_set_view(chunks) if chunk_set is None else chunk_set
        self.chunk_sets[chunk_set.version] = chunk_set
        self.chunks[chunk_set.version] = tuple(chunks)
        return chunk_set

    def _check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def list_chunk_sets(self, chunking_fingerprint: str | None) -> Iterator[models.ChunkSetView]:
        self._check()
        with self._lock:
            self.listings.append(chunking_fingerprint)
        for _, chunk_set in sorted(self.chunk_sets.items()):
            if chunking_fingerprint in (None, chunk_set.chunking_fingerprint):
                yield chunk_set

    def list_chunks(
        self, scene_set_version: str, chunking_fingerprint: str
    ) -> Sequence[models.ChunkView]:
        self._check()
        with self._lock:
            self.fetches.append((scene_set_version, chunking_fingerprint))
        return sorted(
            (
                chunk
                for chunks in self.chunks.values()
                for chunk in chunks
                if chunk.scene_set_version == scene_set_version
                and chunk.chunking_fingerprint == chunking_fingerprint
            ),
            key=lambda chunk: chunk.chunk_index,
        )


class FakeEmbeddingModel:
    """EmbeddingModel with a fixed fingerprint, vectors from the text's hash, or raising."""

    def __init__(
        self,
        fingerprint: str = STEP_FP,
        dimensions: int = DIMENSIONS,
        fail: bool = False,
        model: models.ModelRef = MODEL,
    ) -> None:
        self._fingerprint = fingerprint
        self._dimensions = dimensions
        self._fail = fail
        self._model = model
        self.batches: list[tuple[str, ...]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> models.ModelRef:
        return self._model

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> tuple[models.Vector, ...]:
        with self._lock:
            self.batches.append(tuple(texts))
        if self._fail:
            raise ModelError("model down")
        return tuple(fake_vector(text, self._dimensions) for text in texts)


class FakeSearchIndex:
    """Search Index in dictionaries, with the semantics of the Qdrant backend.

    `tests/embedding/test_search_index.py` runs the same cases against both, so the fake
    stays faithful.
    """

    def __init__(self) -> None:
        self.collections: dict[str, tuple[int, str]] = {}
        self._points: dict[str, dict[str, models.Point]] = {}
        self._lock = threading.Lock()

    def points(self, collection: str) -> dict[str, models.Point]:
        """Return every stored point of the collection, for whole-state assertions."""
        return dict(self._points.get(collection, {}))

    def describe(self, collection: str) -> models.CollectionInfo | None:
        if collection not in self.collections:
            return None
        dimensions, distance = self.collections[collection]
        return models.CollectionInfo(dimensions, distance, len(self._points[collection]))

    def ensure_collection(self, name: str, dimensions: int, distance: str) -> None:
        with self._lock:
            existing = self.collections.get(name)
            if existing is None:
                self.collections[name] = (dimensions, distance)
                self._points[name] = {}
            elif existing != (dimensions, distance):
                raise IndexMismatchError(
                    f"collection {name} holds {existing[0]}-dimensional {existing[1]} vectors"
                )

    def upsert(self, collection: str, points: Sequence[models.Point]) -> None:
        with self._lock:
            store = self._points[collection]
            for point in points:
                store[point.id] = point

    def count(self, collection: str, *, chunk_set_version: str | None = None) -> int:
        return sum(
            1
            for point in self._points[collection].values()
            if chunk_set_version in (None, point.entry.chunk_set_version)
        )

    def get_entries(self, collection: str, point_ids: Sequence[str]) -> Sequence[models.IndexEntry]:
        store = self._points[collection]
        return [store[point_id].entry for point_id in point_ids if point_id in store]

    def drop_collection(self, collection: str) -> None:
        with self._lock:
            self.collections.pop(collection, None)
            self._points.pop(collection, None)


class FakeIndexStore:
    """Index ledger in dictionaries, with the command semantics of the MongoDB backend."""

    def __init__(self) -> None:
        self._index_sets: dict[str, models.IndexSet] = {}
        self._indexes: dict[str, models.Index] = {}
        self._lock = threading.Lock()

    def index_sets(self) -> dict[str, models.IndexSet]:
        """Return every stored index set, for whole-state assertions."""
        return dict(self._index_sets)

    def indexes(self) -> dict[str, models.Index]:
        """Return every stored index record, for whole-state assertions."""
        return dict(self._indexes)

    def get_index_set(self, version: str) -> models.IndexSet | None:
        return self._index_sets.get(version)

    def get_index(self, fingerprint: str) -> models.Index | None:
        return self._indexes.get(fingerprint)

    def list_done(self, fingerprint: str) -> frozenset[str]:
        return frozenset(
            item.chunk_set_version
            for item in self._index_sets.values()
            if item.lineage.fingerprint == fingerprint
        )

    def put_index_set(self, index_set: models.IndexSet) -> bool:
        with self._lock:
            if index_set.version in self._index_sets:
                return False
            self._index_sets[index_set.version] = index_set
            return True

    def put_index(self, index: models.Index) -> bool:
        with self._lock:
            if index.fingerprint in self._indexes:
                return False
            self._indexes[index.fingerprint] = index
            return True


class RecordingWriter:
    """Both writer sides, logging every command and delegating to real ones."""

    def __init__(
        self,
        index: SearchIndexWriter,
        store: IndexStoreWriter,
        fail_after: int | None = None,
    ) -> None:
        self._index = index
        self._store = store
        self._fail_after = fail_after
        self.commands: list[tuple[str, str, int]] = []
        self._lock = threading.Lock()

    def _record(self, command: tuple[str, str, int]) -> None:
        with self._lock:
            self.commands.append(command)
            if self._fail_after is not None and len(self.commands) > self._fail_after:
                raise RuntimeError(f"injected failure in {command[0]}")

    def ensure_collection(self, name: str, dimensions: int, distance: str) -> None:
        self._record(("ensure_collection", name, dimensions))
        self._index.ensure_collection(name, dimensions, distance)

    def upsert(self, collection: str, points: Sequence[models.Point]) -> None:
        self._record(("upsert", points[0].entry.chunk_set_version if points else "", len(points)))
        self._index.upsert(collection, points)

    def put_index_set(self, index_set: models.IndexSet) -> bool:
        self._record(("put_index_set", index_set.chunk_set_version, index_set.vector_count))
        return self._store.put_index_set(index_set)

    def put_index(self, index: models.Index) -> bool:
        self._record(("put_index", index.fingerprint, index.dimensions))
        return self._store.put_index(index)


class FakeChunkSetEmbedder:
    """ChunkSetEmbedder answering per chunk set from a script: an outcome or an exception."""

    def __init__(
        self,
        script: Mapping[str, Outcome | Exception] | None = None,
        fingerprint: str = STEP_FP,
    ) -> None:
        self.script = dict(script or {})
        self._fingerprint = fingerprint
        self.calls: list[str] = []
        self._lock = threading.Lock()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, chunk_set: models.ChunkSetView) -> Outcome:
        with self._lock:
            self.calls.append(chunk_set.version)
        answer = self.script.get(chunk_set.version, "indexed")
        if isinstance(answer, Exception):
            raise answer
        return answer


class RecordingDispatcher:
    """Dispatcher delegating to another one, keeping every batch of items it was handed."""

    def __init__(self, inner: Dispatcher | None = None) -> None:
        self._inner = inner or SerialDispatcher()
        self.batches: list[list[object]] = []

    def map[T, R](self, function: Callable[[T], R], items: Iterable[T]) -> Iterator[R]:
        batch = list(items)
        self.batches.append(list(batch))
        return self._inner.map(function, batch)


type Answer = Json | Exception | Callable[[Json | None], Json]


class FakeTransport:
    """JsonTransport answering from a script keyed by ``METHOD path``, recording every call.

    An answer is a document, an exception to raise, or a function of the payload. A path
    without a script entry answers 404.
    """

    def __init__(self, answers: Mapping[str, Answer] | None = None, base: str = "") -> None:
        self.answers: dict[str, Answer] = dict(answers or {})
        self.base = base
        self.calls: list[tuple[str, str, Json | None, float]] = []
        self._lock = threading.Lock()

    def exchange(self, method: str, url: str, payload: Json | None, timeout: float) -> Json:
        with self._lock:
            self.calls.append((method, url, payload, timeout))
        path = url.removeprefix(self.base)
        answer = self.answers.get(f"{method} {path}")
        if answer is None:
            raise HttpStatusError(404, url, "not found")
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(payload)
        return answer

    def unreachable(self, *paths: str) -> None:
        """Make the given ``METHOD path`` keys fail as if the server were down."""
        for path in paths:
            self.answers[path] = TransportError(f"could not reach {self.base}{path}")


# --- harness -------------------------------------------------------------------------------


@dataclasses.dataclass
class EmbeddingHarness:
    """An embedding pass wired to fakes, exposing them for assertions.

    `service` runs the fake model over the fake Chunk Store into the fake Search Index and
    ledger; `scheduler` is the pass over it. `variant_service` and `variant` rebuild them
    with some collaborators replaced.
    """

    reader: FakeChunkReader
    model: FakeEmbeddingModel
    search_index: FakeSearchIndex
    store: FakeIndexStore
    writer: RecordingWriter
    clock: FakeClock
    config: EmbeddingConfig
    index: models.Index
    service: EmbeddingService
    scheduler: Scheduler

    def seed(self, *chunk_sets: Sequence[models.ChunkView]) -> None:
        """Register the chunks of each chunk set with the fake Chunk Store."""
        for chunks in chunk_sets:
            self.reader.seed(chunks)

    def variant_service(
        self,
        *,
        model: EmbeddingModel | None = None,
        writer: RecordingWriter | None = None,
        config: EmbeddingConfig | None = None,
        batch_size: int | None = None,
        index: models.Index | None = None,
    ) -> EmbeddingService:
        writer = self.writer if writer is None else writer
        return EmbeddingService(
            reader=self.reader,
            model=self.model if model is None else model,
            index_writer=writer,
            store_writer=writer,
            index=self.index if index is None else index,
            clock=self.clock,
            config=self.config if config is None else config,
            batch_size=2 if batch_size is None else batch_size,
        )

    def variant(
        self,
        *,
        dispatcher: Dispatcher | None = None,
        service: EmbeddingService | None = None,
        config: EmbeddingConfig | None = None,
    ) -> Scheduler:
        return Scheduler(
            chunk_reader=self.reader,
            store_reader=self.store,
            embedder=self.service if service is None else service,
            dispatcher=dispatcher or SerialDispatcher(),
            config=self.config if config is None else config,
        )


def make_embedding_harness(
    *chunk_sets: Sequence[models.ChunkView],
    dispatcher: Dispatcher | None = None,
    config: EmbeddingConfig | None = None,
    batch_size: int = 2,
) -> EmbeddingHarness:
    config = EmbeddingConfig() if config is None else config
    reader = FakeChunkReader()
    model = FakeEmbeddingModel()
    search_index = FakeSearchIndex()
    store = FakeIndexStore()
    writer = RecordingWriter(search_index, store)
    clock = FakeClock()
    index = prepare_index(
        model=model,
        index_writer=search_index,
        store_reader=store,
        store_writer=store,
        config=config,
        clock=clock,
    )
    service = EmbeddingService(
        reader=reader,
        model=model,
        index_writer=writer,
        store_writer=writer,
        index=index,
        clock=clock,
        config=config,
        batch_size=batch_size,
    )
    scheduler = Scheduler(
        chunk_reader=reader,
        store_reader=store,
        embedder=service,
        dispatcher=dispatcher or SerialDispatcher(),
        config=config,
    )
    harness = EmbeddingHarness(
        reader, model, search_index, store, writer, clock, config, index, service, scheduler
    )
    harness.seed(*chunk_sets)
    return harness


def as_document(value: Any) -> Any:  # noqa: ANN401 - walks untyped JSON
    """Render a store document the way the samples are written: dates as RFC 3339 strings."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: as_document(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_document(v) for v in value]
    return value
