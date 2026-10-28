"""Fakes and builders for the tests of the chunking step."""

import dataclasses
import hashlib
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import ChunkingConfig
from scene_rag.retrieval.chunking.dispatch import SerialDispatcher
from scene_rag.retrieval.chunking.per_scene import PerSceneChunker
from scene_rag.retrieval.chunking.protocols import (
    Chunker,
    ChunkStoreWriter,
    Dispatcher,
    SceneApiError,
)
from scene_rag.retrieval.chunking.report import Outcome
from scene_rag.retrieval.chunking.scheduler import Scheduler
from scene_rag.retrieval.chunking.service import ChunkingService

T0 = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
SCENE_BOUNDS: tuple[tuple[float, float], ...] = ((0.0, 4.0), (4.0, 8.0), (8.0, 12.5))
FPS = 25.0
SEGMENTATION_FP = "3b9e1f6a2c40"
DESCRIPTION_FP = "0123456789ab"
TRANSCRIPT_FP = "e41a9c27b0d5"
PUBLIC = frozenset({"public"})
STEP_FP = "c7a1e2f3b4d5"

# JSON documents are untyped; `Any` is confined to the encoders below.
type Json = dict[str, Any]


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


def make_source(
    asset_id: str = "asset-0000000000000001", source_id: str = "local", filename: str = "clip.mp4"
) -> models.SourceView:
    return models.SourceView(source_id=source_id, asset_id=asset_id, filename=filename)


def make_asset(
    *sources: models.SourceView, entitlements: frozenset[str] = PUBLIC
) -> models.AssetView:
    return models.AssetView(
        media_type="video/mp4",
        size_bytes=4,
        entitlements=entitlements,
        sources=sources or (make_source(),),
    )


def make_media(duration: float = 12.5) -> models.MediaView:
    return models.MediaView(
        duration_seconds=duration,
        frame_rate=FPS,
        width=1920,
        height=1080,
        frame_count=round(duration * FPS),
    )


def make_derivative(
    data: bytes = b"jpeg", media_type: str = "image/jpeg"
) -> models.DerivativeRefView:
    digest = sha256_version(data)
    return models.DerivativeRefView(
        digest=digest, media_type=media_type, size_bytes=len(data), url=f"/derivatives/{digest}"
    )


def make_scene(index: int, bounds: tuple[float, float], *, keyframes: int = 2) -> models.SceneView:
    start, end = bounds
    return models.SceneView(
        index=index,
        start_seconds=start,
        end_seconds=end,
        start_frame=round(start * FPS),
        end_frame=round(end * FPS),
        keyframes=tuple(
            models.KeyframeView(
                derivative=make_derivative(f"jpeg:{index}:{pick}".encode()),
                frame=round(start * FPS) + pick * 10,
                seconds=start + pick * 0.4,
            )
            for pick in range(keyframes)
        ),
        audio_clip=make_derivative(f"wav:{index}".encode(), "audio/wav"),
    )


def scene_set_version(asset_version: str, fingerprint: str = SEGMENTATION_FP) -> str:
    return sha256_version(f"{asset_version}\n{fingerprint}".encode())


def enrichment_version(scene_set: str, index: int, enricher: str, fingerprint: str) -> str:
    return sha256_version(f"{scene_set}\n{index}\n{enricher}\n{fingerprint}".encode())


def make_enrichment(
    scene_set: str,
    scene: models.SceneView,
    enricher: str = "description",
    fingerprint: str = DESCRIPTION_FP,
    segments: tuple[models.SegmentView, ...] | None = None,
    classification: str = "internal",
) -> models.EnrichmentView:
    if segments is None:
        segments = (
            models.SegmentView(
                scene.start_seconds, scene.end_seconds, f"{enricher} of scene {scene.index}"
            ),
        )
    return models.EnrichmentView(
        version=enrichment_version(scene_set, scene.index, enricher, fingerprint),
        enricher=enricher,
        fingerprint=fingerprint,
        classification=classification,
        model=models.ModelRef(f"{enricher}-model", "sha256:" + "ab" * 32),
        segments=segments,
    )


def scene_record_version(scene_set: str, index: int, enrichment_versions: Iterable[str]) -> str:
    lines = [scene_set, str(index), *sorted(enrichment_versions)]
    return sha256_version("\n".join(lines).encode())


def make_record(
    scene: models.SceneView,
    enrichments: tuple[models.EnrichmentView, ...],
    *,
    asset_version: str = sha256_version(b"abcd"),
    segmentation_fingerprint: str = SEGMENTATION_FP,
    asset: models.AssetView | None = None,
    live: bool = True,
) -> models.SceneRecordView:
    scene_set = scene_set_version(asset_version, segmentation_fingerprint)
    return models.SceneRecordView(
        version=scene_record_version(scene_set, scene.index, (e.version for e in enrichments)),
        scene_set_version=scene_set,
        asset_version=asset_version,
        segmentation_fingerprint=segmentation_fingerprint,
        live=live,
        asset=make_asset() if asset is None else asset,
        media=make_media(),
        scene=scene,
        enrichments=tuple(sorted(enrichments, key=lambda e: e.pin)),
    )


def make_records(
    asset_version: str = sha256_version(b"abcd"),
    *,
    segmentation_fingerprint: str = SEGMENTATION_FP,
    bounds: Sequence[tuple[float, float]] = SCENE_BOUNDS,
    enrichers: Mapping[str, str] | None = None,
    asset: models.AssetView | None = None,
    live: bool = True,
) -> tuple[models.SceneRecordView, ...]:
    """Scene records of one scene set, every scene enriched by every enricher given.

    `enrichers` maps enricher name to fingerprint; description and transcript by default.
    """
    pins = (
        {"description": DESCRIPTION_FP, "transcript": TRANSCRIPT_FP}
        if enrichers is None
        else enrichers
    )
    scene_set = scene_set_version(asset_version, segmentation_fingerprint)
    records = []
    for index, span in enumerate(bounds):
        scene = make_scene(index, span)
        enrichments = tuple(
            make_enrichment(scene_set, scene, name, fp) for name, fp in pins.items()
        )
        records.append(
            make_record(
                scene,
                enrichments,
                asset_version=asset_version,
                segmentation_fingerprint=segmentation_fingerprint,
                asset=asset,
                live=live,
            )
        )
    return tuple(records)


def summary_of(
    records: Sequence[models.SceneRecordView], *, live: bool | None = None
) -> models.SceneSetSummaryView:
    """The summary scene-api would list for the records of one scene set."""
    first = records[0]
    coverage: dict[tuple[str, str], int] = {}
    for record in records:
        for item in record.enrichments:
            key = (item.enricher, item.fingerprint)
            coverage[key] = coverage.get(key, 0) + 1
    return models.SceneSetSummaryView(
        version=first.scene_set_version,
        asset_version=first.asset_version,
        segmentation_fingerprint=first.segmentation_fingerprint,
        scene_count=len(records),
        created_at=T0,
        live=first.live if live is None else live,
        enrichments=tuple(
            models.CoverageView(enricher, fingerprint, count, count >= len(records))
            for (enricher, fingerprint), count in sorted(coverage.items())
        ),
    )


def make_summary(
    asset_version: str = sha256_version(b"abcd"),
    *,
    scene_count: int = 3,
    coverage: Sequence[tuple[str, str, bool]] = (
        ("description", DESCRIPTION_FP, True),
        ("transcript", TRANSCRIPT_FP, True),
    ),
    live: bool = True,
) -> models.SceneSetSummaryView:
    return models.SceneSetSummaryView(
        version=scene_set_version(asset_version),
        asset_version=asset_version,
        segmentation_fingerprint=SEGMENTATION_FP,
        scene_count=scene_count,
        created_at=T0,
        live=live,
        enrichments=tuple(
            models.CoverageView(name, fp, scene_count if complete else scene_count - 1, complete)
            for name, fp, complete in coverage
        ),
    )


def make_chunk(
    records: Sequence[models.SceneRecordView] | None = None,
    index: int = 0,
    fingerprint: str = STEP_FP,
    text: str | None = None,
    created_at: datetime = T0,
) -> models.Chunk:
    records = make_records() if records is None else records
    record = records[index]
    version = models.chunk_version([record.version], fingerprint, index, "sha256")
    return models.Chunk(
        version=version,
        scene_set_version=record.scene_set_version,
        asset_version=record.asset_version,
        segmentation_fingerprint=record.segmentation_fingerprint,
        chunk_index=index,
        scene_indices=(record.scene.index,),
        start_seconds=record.scene.start_seconds,
        end_seconds=record.scene.end_seconds,
        text=f"chunk {index}" if text is None else text,
        sources=record.asset.sources,
        entitlements=record.asset.entitlements,
        classifications=("internal",),
        keyframes=tuple(
            models.KeyframeRef(k.derivative.digest, k.seconds) for k in record.scene.keyframes
        ),
        enrichments=tuple(e.ref for e in record.enrichments),
        created_at=created_at,
        lineage=models.Lineage(
            fingerprint=fingerprint,
            inputs=(models.SCENE_RECORD_INPUT_PREFIX + record.version,),
            version=version,
        ),
    )


def make_chunk_set(
    records: Sequence[models.SceneRecordView] | None = None,
    fingerprint: str = STEP_FP,
    chunk_count: int | None = None,
    created_at: datetime = T0,
) -> models.ChunkSet:
    records = make_records() if records is None else records
    first = records[0]
    version = models.chunk_set_version(first.scene_set_version, fingerprint, "sha256")
    return models.ChunkSet(
        version=version,
        scene_set_version=first.scene_set_version,
        asset_version=first.asset_version,
        segmentation_fingerprint=first.segmentation_fingerprint,
        pins=tuple(sorted({e.pin for e in first.enrichments})),
        chunk_count=len(records) if chunk_count is None else chunk_count,
        created_at=created_at,
        lineage=models.Lineage(
            fingerprint=fingerprint,
            inputs=tuple(models.SCENE_RECORD_INPUT_PREFIX + r.version for r in records),
            version=version,
        ),
    )


# --- JSON encoders of the views, mirroring extraction's scene_api/encoding.py -------------


def encode_datetime(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def encode_derivative(reference: models.DerivativeRefView) -> Json:
    return {
        "digest": reference.digest,
        "media_type": reference.media_type,
        "size_bytes": reference.size_bytes,
        "url": reference.url,
    }


def encode_scene_record(record: models.SceneRecordView) -> Json:
    scene = record.scene
    return {
        "schema_version": 1,
        "version": record.version,
        "scene_set_version": record.scene_set_version,
        "asset_version": record.asset_version,
        "segmentation_fingerprint": record.segmentation_fingerprint,
        "live": record.live,
        "asset": {
            "media_type": record.asset.media_type,
            "size_bytes": record.asset.size_bytes,
            "entitlements": sorted(record.asset.entitlements),
            "sources": [dataclasses.asdict(source) for source in record.asset.sources],
        },
        "media": dataclasses.asdict(record.media),
        "scene": {
            "index": scene.index,
            "start_seconds": scene.start_seconds,
            "end_seconds": scene.end_seconds,
            "start_frame": scene.start_frame,
            "end_frame": scene.end_frame,
            "keyframes": [
                {**encode_derivative(k.derivative), "frame": k.frame, "seconds": k.seconds}
                for k in scene.keyframes
            ],
            "audio_clip": None if scene.audio_clip is None else encode_derivative(scene.audio_clip),
        },
        "enrichments": [
            {
                "version": item.version,
                "enricher": item.enricher,
                "fingerprint": item.fingerprint,
                "classification": item.classification,
                "model": {"name": item.model.name, "digest": item.model.digest},
                "segments": [dataclasses.asdict(segment) for segment in item.segments],
            }
            for item in record.enrichments
        ],
    }


def encode_summary(summary: models.SceneSetSummaryView) -> Json:
    return {
        "version": summary.version,
        "asset_version": summary.asset_version,
        "segmentation_fingerprint": summary.segmentation_fingerprint,
        "scene_count": summary.scene_count,
        "created_at": encode_datetime(summary.created_at),
        "live": summary.live,
        "enrichments": [dataclasses.asdict(item) for item in summary.enrichments],
    }


def encode_asset_version(asset: models.AssetVersionView) -> Json:
    return {
        "version": asset.version,
        "live": asset.live,
        "retired_at": None if asset.live else encode_datetime(T0),
        "media_type": "video/mp4",
        "size_bytes": 4,
        "entitlements": sorted(PUBLIC) if asset.live else [],
        "sources": [dataclasses.asdict(make_source())] if asset.live else [],
    }


def encode_page(items: Sequence[Json], next_cursor: str | None) -> Json:
    return {"items": list(items), "next": next_cursor}


# --- fakes ---------------------------------------------------------------------------------


class FakeSceneRecordReader:
    """SceneRecordReader over dictionaries of summaries, records and asset liveness.

    Listing honours the query the way scene-api does: live sets only, the segmentation
    filter, and completeness under every pin; records are narrowed to the pins.
    """

    def __init__(self) -> None:
        self.summaries: dict[str, models.SceneSetSummaryView] = {}
        self.records: dict[str, tuple[models.SceneRecordView, ...]] = {}
        self.assets: dict[str, bool] = {}
        self.listings: list[models.SceneSetQuery] = []
        self.fetches: list[tuple[str, tuple[models.EnrichmentPin, ...]]] = []
        self.liveness_checks: list[str] = []
        self.failure: Exception | None = None
        self._lock = threading.Lock()

    def seed(self, records: Sequence[models.SceneRecordView], *, live: bool = True) -> None:
        summary = summary_of(records, live=live)
        self.summaries[summary.version] = summary
        self.records[summary.version] = tuple(records)
        self.assets[summary.asset_version] = live

    def retire(self, asset_version: str) -> None:
        self.assets[asset_version] = False
        for version, summary in list(self.summaries.items()):
            if summary.asset_version == asset_version:
                self.summaries[version] = dataclasses.replace(summary, live=False)
                self.records[version] = tuple(
                    dataclasses.replace(r, live=False) for r in self.records[version]
                )

    def _check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def list_scene_sets(self, query: models.SceneSetQuery) -> Iterator[models.SceneSetSummaryView]:
        self._check()
        with self._lock:
            self.listings.append(query)
        for _, summary in sorted(self.summaries.items()):
            complete = {(c.enricher, c.fingerprint) for c in summary.enrichments if c.complete}
            if (
                self.assets.get(summary.asset_version, False)
                and (
                    query.segmentation_fingerprint is None
                    or summary.segmentation_fingerprint == query.segmentation_fingerprint
                )
                and all((pin.enricher, pin.fingerprint) in complete for pin in query.pins)
            ):
                yield summary

    def list_scene_records(
        self, scene_set_version: str, pins: Sequence[models.EnrichmentPin]
    ) -> Sequence[models.SceneRecordView]:
        self._check()
        with self._lock:
            self.fetches.append((scene_set_version, tuple(pins)))
        if scene_set_version not in self.records:
            raise SceneApiError(f"no scene-set {scene_set_version}")
        wanted = set(pins)
        return [
            dataclasses.replace(
                record,
                enrichments=tuple(e for e in record.enrichments if not wanted or e.pin in wanted),
            )
            for record in self.records[scene_set_version]
        ]

    def get_asset_version(self, version: str) -> models.AssetVersionView | None:
        self._check()
        with self._lock:
            self.liveness_checks.append(version)
        live = self.assets.get(version)
        return None if live is None else models.AssetVersionView(version, live)


class FakeChunkStore:
    """Chunk Store in dictionaries, with the command semantics of the MongoDB backend.

    `tests/chunking/test_chunk_store.py` runs the same cases against both, so the fake stays
    faithful.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, models.Chunk] = {}
        self._chunk_sets: dict[str, models.ChunkSet] = {}
        self._lock = threading.Lock()

    def get_chunk(self, version: str) -> models.Chunk | None:
        return self._chunks.get(version)

    def get_chunk_set(self, version: str) -> models.ChunkSet | None:
        return self._chunk_sets.get(version)

    def chunks(self) -> dict[str, models.Chunk]:
        """Return every stored chunk, for whole-state assertions."""
        return dict(self._chunks)

    def chunk_sets(self) -> dict[str, models.ChunkSet]:
        """Return every stored chunk set, for whole-state assertions."""
        return dict(self._chunk_sets)

    def list_done(self, fingerprint: str) -> frozenset[str]:
        return frozenset(
            item.scene_set_version
            for item in self._chunk_sets.values()
            if item.lineage.fingerprint == fingerprint
        )

    def put_chunk(self, chunk: models.Chunk) -> bool:
        with self._lock:
            if chunk.version in self._chunks:
                return False
            self._chunks[chunk.version] = chunk
            return True

    def put_chunk_set(self, chunk_set: models.ChunkSet) -> bool:
        with self._lock:
            if chunk_set.version in self._chunk_sets:
                return False
            self._chunk_sets[chunk_set.version] = chunk_set
            return True


class RecordingChunkWriter:
    """ChunkStoreWriter that logs every command, delegating to a real writer."""

    def __init__(self, inner: ChunkStoreWriter, fail_after: int | None = None) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self.commands: list[tuple[str, str, int]] = []
        self._lock = threading.Lock()

    def _record(self, command: tuple[str, str, int]) -> None:
        with self._lock:
            self.commands.append(command)
            if self._fail_after is not None and len(self.commands) > self._fail_after:
                raise RuntimeError(f"injected failure in {command[0]}")

    def put_chunk(self, chunk: models.Chunk) -> bool:
        self._record(("put_chunk", chunk.scene_set_version, chunk.chunk_index))
        return self._inner.put_chunk(chunk)

    def put_chunk_set(self, chunk_set: models.ChunkSet) -> bool:
        self._record(("put_chunk_set", chunk_set.scene_set_version, chunk_set.chunk_count))
        return self._inner.put_chunk_set(chunk_set)


class FakeChunker:
    """Chunker with a fixed fingerprint, drafting one chunk per record, or raising."""

    def __init__(self, fingerprint: str = STEP_FP, fail: bool = False) -> None:
        self._fingerprint = fingerprint
        self._fail = fail
        self.calls: list[tuple[str, ...]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "fake"

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def chunk(self, records: Sequence[models.SceneRecordView]) -> tuple[models.ChunkDraft, ...]:
        with self._lock:
            self.calls.append(tuple(r.version for r in records))
        if self._fail:
            raise RuntimeError("chunker down")
        return tuple(
            models.ChunkDraft(
                scene_indices=(r.scene.index,),
                start_seconds=r.scene.start_seconds,
                end_seconds=r.scene.end_seconds,
                text=f"fake chunk {r.scene.index}",
                enrichments=tuple(e.ref for e in r.enrichments),
            )
            for r in records
        )


class FakeSceneSetChunker:
    """SceneSetChunker answering per scene set from a script: an outcome or an exception."""

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

    def chunk(self, summary: models.SceneSetSummaryView) -> Outcome:
        with self._lock:
            self.calls.append(summary.version)
        answer = self.script.get(summary.version, "chunked")
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


@dataclasses.dataclass
class ChunkingHarness:
    """A chunking pass wired to fakes, exposing them for assertions.

    `service` runs the real `PerSceneChunker` over the fake scene-api; `scheduler` is the
    pass over it. `variant_service` and `variant` rebuild them with some collaborators
    replaced.
    """

    reader: FakeSceneRecordReader
    store: FakeChunkStore
    writer: RecordingChunkWriter
    clock: FakeClock
    config: ChunkingConfig
    chunker: PerSceneChunker
    service: ChunkingService
    scheduler: Scheduler

    def seed(self, *scene_sets: Sequence[models.SceneRecordView], live: bool = True) -> None:
        """Register the scene records of each scene set with the fake scene-api."""
        for records in scene_sets:
            self.reader.seed(records, live=live)

    def variant_service(
        self,
        *,
        chunker: Chunker | None = None,
        writer: ChunkStoreWriter | None = None,
        config: ChunkingConfig | None = None,
    ) -> ChunkingService:
        return ChunkingService(
            reader=self.reader,
            writer=writer or self.writer,
            chunker=chunker or self.chunker,
            clock=self.clock,
            config=config or self.config,
        )

    def variant(
        self,
        *,
        dispatcher: Dispatcher | None = None,
        service: ChunkingService | None = None,
        config: ChunkingConfig | None = None,
    ) -> Scheduler:
        return Scheduler(
            reader=self.reader,
            chunk_reader=self.store,
            chunker=service or self.service,
            dispatcher=dispatcher or SerialDispatcher(),
            config=config or self.config,
        )


def make_chunking_harness(
    *scene_sets: Sequence[models.SceneRecordView],
    dispatcher: Dispatcher | None = None,
    config: ChunkingConfig | None = None,
) -> ChunkingHarness:
    config = ChunkingConfig() if config is None else config
    reader = FakeSceneRecordReader()
    store = FakeChunkStore()
    writer = RecordingChunkWriter(store)
    clock = FakeClock()
    chunker = PerSceneChunker(config)
    service = ChunkingService(
        reader=reader, writer=writer, chunker=chunker, clock=clock, config=config
    )
    scheduler = Scheduler(
        reader=reader,
        chunk_reader=store,
        chunker=service,
        dispatcher=dispatcher or SerialDispatcher(),
        config=config,
    )
    harness = ChunkingHarness(reader, store, writer, clock, config, chunker, service, scheduler)
    harness.seed(*scene_sets)
    return harness
