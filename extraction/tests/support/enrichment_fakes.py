"""Fakes and builders for the tests of the enrichment step."""

import dataclasses
import hashlib
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import EnrichmentConfig
from scene_rag.extraction.enrichment.description import DescriptionEnricher
from scene_rag.extraction.enrichment.dispatch import SerialDispatcher
from scene_rag.extraction.enrichment.protocols import (
    DerivativeUnavailableError,
    Dispatcher,
    Enricher,
    EnrichmentStoreWriter,
    ModelError,
    Transcript,
    TranscriptSegment,
)
from scene_rag.extraction.enrichment.scheduler import Scheduler
from scene_rag.extraction.enrichment.service import EnrichmentService
from scene_rag.extraction.enrichment.transcription import TranscriptionEnricher

T0 = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
SCENE_BOUNDS: tuple[tuple[float, float], ...] = ((0.0, 4.0), (4.0, 8.0), (8.0, 12.5))
VISION_MODEL = models.ModelRef("qwen2.5vl:3b", "sha256:" + "a1" * 32)
SPEECH_MODEL = models.ModelRef("small", "sha256:" + "b2" * 32)


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


def make_asset_view(
    version: str = sha256_version(b"abcd"),
    *,
    has_sources: bool = True,
    retired_at: datetime | None = None,
) -> models.AssetVersionView:
    return models.AssetVersionView(version=version, has_sources=has_sources, retired_at=retired_at)


def make_derivative(
    data: bytes = b"jpeg", media_type: str = "image/jpeg"
) -> models.DerivativeRefView:
    digest = hashlib.sha256(data).hexdigest()
    extension = "jpg" if media_type == "image/jpeg" else "wav"
    return models.DerivativeRefView(
        digest=f"sha256:{digest}",
        uri=f"file:///derivatives/sha256/{digest[:2]}/{digest}.{extension}",
        media_type=media_type,
        size_bytes=len(data),
    )


def keyframe_bytes(scene_index: int, pick: int) -> bytes:
    return f"jpeg:{scene_index}:{pick}".encode()


def clip_bytes(scene_index: int) -> bytes:
    return f"wav:{scene_index}".encode()


def make_scene(
    index: int,
    bounds: tuple[float, float],
    *,
    keyframes: int = 2,
    audio: bool = True,
) -> models.SceneView:
    start, end = bounds
    return models.SceneView(
        index=index,
        start_seconds=start,
        end_seconds=end,
        keyframes=tuple(
            models.KeyframeView(
                derivative=make_derivative(keyframe_bytes(index, pick)),
                frame=round(start * 25) + pick * 10,
                seconds=start + pick * 0.4,
            )
            for pick in range(keyframes)
        ),
        audio_clip=make_derivative(clip_bytes(index), "audio/wav") if audio else None,
    )


def make_scene_set(
    asset_version: str = sha256_version(b"abcd"),
    *,
    segmentation_fingerprint: str = "3b9e1f6a2c40",
    bounds: Sequence[tuple[float, float]] = SCENE_BOUNDS,
    keyframes: int = 2,
    audio: bool = True,
) -> models.SceneSetView:
    digest = hashlib.sha256(f"{asset_version}\n{segmentation_fingerprint}".encode()).hexdigest()
    return models.SceneSetView(
        version=f"sha256:{digest}",
        asset_version=asset_version,
        scenes=tuple(
            make_scene(index, span, keyframes=keyframes, audio=audio)
            for index, span in enumerate(bounds)
        ),
    )


def summary_of(scene_set: models.SceneSetView) -> models.SceneSetSummary:
    return models.SceneSetSummary(
        version=scene_set.version,
        asset_version=scene_set.asset_version,
        scene_count=len(scene_set.scenes),
    )


def blobs_of(*scene_sets: models.SceneSetView) -> dict[str, bytes]:
    """The derivative bytes every scene of the sets references, by digest."""
    blobs: dict[str, bytes] = {}
    for scene_set in scene_sets:
        for scene in scene_set.scenes:
            for pick, keyframe in enumerate(scene.keyframes):
                blobs[keyframe.derivative.digest] = keyframe_bytes(scene.index, pick)
            if scene.audio_clip is not None:
                blobs[scene.audio_clip.digest] = clip_bytes(scene.index)
    return blobs


def make_enrichment(
    scene_set: models.SceneSetView | None = None,
    scene_index: int = 0,
    enricher: str = "description",
    fingerprint: str = "0123456789ab",
    segments: tuple[models.SemanticSegment, ...] | None = None,
    created_at: datetime = T0,
) -> models.Enrichment:
    scene_set = make_scene_set() if scene_set is None else scene_set
    scene = scene_set.scenes[scene_index]
    if segments is None:
        segments = (
            models.SemanticSegment(scene.start_seconds, scene.end_seconds, f"scene {scene_index}"),
        )
    version = models.enrichment_version(
        scene_set.version, scene_index, enricher, fingerprint, "sha256"
    )
    return models.Enrichment(
        version=version,
        scene_set_version=scene_set.version,
        asset_version=scene_set.asset_version,
        scene_index=scene_index,
        enricher=enricher,
        classification="internal",
        model=VISION_MODEL if enricher == "description" else SPEECH_MODEL,
        segments=segments,
        created_at=created_at,
        lineage=models.Lineage(
            fingerprint=fingerprint,
            inputs=(
                models.SCENE_SET_INPUT_PREFIX + scene_set.version,
                models.DERIVATIVE_INPUT_PREFIX + scene.keyframes[0].derivative.digest,
            ),
            version=version,
        ),
    )


class FakeAssetVersionReader:
    """AssetVersionReader over a mutable dictionary of views."""

    def __init__(self, *records: models.AssetVersionView) -> None:
        self.records: dict[str, models.AssetVersionView] = {r.version: r for r in records}

    def get(self, version: str) -> models.AssetVersionView | None:
        return self.records.get(version)

    def list_live_versions(self) -> frozenset[str]:
        return frozenset(v for v, record in self.records.items() if record.is_live)

    def retire(self, version: str, at: datetime = T0) -> None:
        self.records[version] = dataclasses.replace(
            self.records[version], has_sources=False, retired_at=at
        )


class FakeSceneSetReader:
    """SceneSetReader over a mutable dictionary of scene sets."""

    def __init__(self, *scene_sets: models.SceneSetView) -> None:
        self.scene_sets: dict[str, models.SceneSetView] = {s.version: s for s in scene_sets}

    def get(self, version: str) -> models.SceneSetView | None:
        return self.scene_sets.get(version)

    def list_summaries(self) -> Sequence[models.SceneSetSummary]:
        return [summary_of(scene_set) for _, scene_set in sorted(self.scene_sets.items())]


class FakeEnrichmentStore:
    """Enrichment Store in a dictionary, with the command semantics of the MongoDB backend.

    `tests/enrichment/test_enrichment_store.py` runs the same cases against both, so the fake
    stays faithful.
    """

    def __init__(self) -> None:
        self._records: dict[str, models.Enrichment] = {}
        self._lock = threading.Lock()

    def get(self, version: str) -> models.Enrichment | None:
        return self._records.get(version)

    def snapshot(self) -> dict[str, models.Enrichment]:
        """Return every stored record, for whole-state assertions."""
        return dict(self._records)

    def list_done(self, enricher: str, fingerprint: str) -> frozenset[tuple[str, int]]:
        return frozenset(
            (record.scene_set_version, record.scene_index)
            for record in self._records.values()
            if record.enricher == enricher and record.lineage.fingerprint == fingerprint
        )

    def put(self, record: models.Enrichment) -> bool:
        with self._lock:
            if record.version in self._records:
                return False
            self._records[record.version] = record
            return True


class RecordingEnrichmentWriter:
    """EnrichmentStoreWriter that logs every command, delegating to a real writer."""

    def __init__(self, inner: EnrichmentStoreWriter, fail_after: int | None = None) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self.commands: list[tuple[str, str, int]] = []
        self._lock = threading.Lock()

    def put(self, record: models.Enrichment) -> bool:
        with self._lock:
            self.commands.append(("put", record.enricher, record.scene_index))
            if self._fail_after is not None and len(self.commands) > self._fail_after:
                raise RuntimeError("injected failure in put")
        return self._inner.put(record)


class FakeDerivativeReader:
    """DerivativeReader over bytes by digest; digests not held are unavailable."""

    def __init__(self, blobs: Mapping[str, bytes] | None = None) -> None:
        self.blobs: dict[str, bytes] = dict(blobs or {})
        self.reads: list[str] = []
        self._lock = threading.Lock()

    def read(self, reference: models.DerivativeRefView) -> bytes:
        with self._lock:
            self.reads.append(reference.digest)
        try:
            return self.blobs[reference.digest]
        except KeyError:
            raise DerivativeUnavailableError(f"{reference.uri}: not filed") from None


class FakeVisionModel:
    """VisionModel answering with a text derived from the images, or a fixed one, or failing."""

    def __init__(
        self,
        answer: str | Callable[[Sequence[bytes], str], str] | None = None,
        fail: bool = False,
    ) -> None:
        self._answer = answer
        self._fail = fail
        self.calls: list[tuple[tuple[bytes, ...], str]] = []
        self._lock = threading.Lock()

    @property
    def model(self) -> models.ModelRef:
        return VISION_MODEL

    def describe(self, images: Sequence[bytes], prompt: str) -> str:
        with self._lock:
            self.calls.append((tuple(images), prompt))
        if self._fail:
            raise ModelError("vision model down")
        if self._answer is None:
            return "Seen: " + ", ".join(image.decode() for image in images)
        if isinstance(self._answer, str):
            return self._answer
        return self._answer(images, prompt)


def heard(text: str, start: float, end: float, confidence: float | None = 0.9) -> TranscriptSegment:
    return TranscriptSegment(start_seconds=start, end_seconds=end, text=text, confidence=confidence)


class FakeTranscriber:
    """Transcriber answering a fixed transcript, or one derived from the clip, or failing."""

    def __init__(self, transcript: Transcript | None = None, fail: bool = False) -> None:
        self._transcript = transcript
        self._fail = fail
        self.calls: list[bytes] = []
        self._lock = threading.Lock()

    @property
    def model(self) -> models.ModelRef:
        return SPEECH_MODEL

    def transcribe(self, audio: bytes) -> Transcript:
        with self._lock:
            self.calls.append(audio)
        if self._fail:
            raise ModelError("speech model down")
        if self._transcript is not None:
            return self._transcript
        label = audio.decode(errors="replace")
        return Transcript(language="en", segments=(heard(f"Heard {label}", 0.5, 1.5),))


class FakeEnricher:
    """Enricher with a fixed name and fingerprint, answering one segment per scene or raising."""

    def __init__(
        self,
        name: str = "description",
        fingerprint: str = "0123456789ab",
        failing: Mapping[tuple[str, int], Exception] | None = None,
    ) -> None:
        self._name = name
        self._fingerprint = fingerprint
        self._failing = dict(failing or {})
        self.calls: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def classification(self) -> str:
        return "internal"

    @property
    def model(self) -> models.ModelRef:
        return VISION_MODEL

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def enrich(
        self, scene_set: models.SceneSetView, scene: models.SceneView
    ) -> models.EnricherOutput:
        key = (scene_set.version, scene.index)
        with self._lock:
            self.calls.append(key)
        if key in self._failing:
            raise self._failing[key]
        return models.EnricherOutput(
            segments=(
                models.SemanticSegment(
                    scene.start_seconds, scene.end_seconds, f"{self._name} of scene {scene.index}"
                ),
            ),
            inputs=tuple(k.derivative.digest for k in scene.keyframes),
        )


class FakeSceneSetEnricher:
    """SceneSetEnricher answering per scene set from a script: an outcome or an exception."""

    def __init__(
        self,
        script: Mapping[str, models.Outcome | Exception] | None = None,
        fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        self.script = dict(script or {})
        self._fingerprints = dict(fingerprints or {"description": "0123456789ab"})
        self.calls: list[tuple[str, dict[str, frozenset[int]]]] = []
        self._lock = threading.Lock()

    @property
    def fingerprints(self) -> Mapping[str, str]:
        return self._fingerprints

    def enrich(
        self, summary: models.SceneSetSummary, pending: Mapping[str, frozenset[int]]
    ) -> models.Outcome:
        with self._lock:
            self.calls.append((summary.version, dict(pending)))
        answer = self.script.get(summary.version, "enriched")
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
class EnrichmentHarness:
    """An enrichment pass wired to fakes, exposing them for assertions.

    `service` runs the real `DescriptionEnricher` and `TranscriptionEnricher` over fake models;
    `scheduler` is the pass over it. `variant_service` and `variant` rebuild them with some
    collaborators replaced.
    """

    assets: FakeAssetVersionReader
    scenes: FakeSceneSetReader
    store: FakeEnrichmentStore
    writer: RecordingEnrichmentWriter
    derivatives: FakeDerivativeReader
    vision: FakeVisionModel
    transcriber: FakeTranscriber
    clock: FakeClock
    config: EnrichmentConfig
    description: DescriptionEnricher
    transcription: TranscriptionEnricher
    service: EnrichmentService
    scheduler: Scheduler

    def seed(self, *scene_sets: models.SceneSetView, live: bool = True) -> None:
        """Register the scene sets, their asset versions and their derivative bytes."""
        for scene_set in scene_sets:
            self.scenes.scene_sets[scene_set.version] = scene_set
            self.assets.records.setdefault(
                scene_set.asset_version, make_asset_view(scene_set.asset_version)
            )
            if not live:
                self.assets.retire(scene_set.asset_version)
        self.derivatives.blobs.update(blobs_of(*scene_sets))

    def variant_service(
        self,
        *,
        enrichers: Sequence[Enricher] | None = None,
        writer: EnrichmentStoreWriter | None = None,
        derivatives: FakeDerivativeReader | None = None,
        vision: FakeVisionModel | None = None,
        transcriber: FakeTranscriber | None = None,
    ) -> EnrichmentService:
        derivatives = derivatives or self.derivatives
        if enrichers is None:
            enrichers = (
                DescriptionEnricher(
                    vision=vision or self.vision,
                    derivatives=derivatives,
                    config=self.config.description,
                ),
                TranscriptionEnricher(
                    transcriber=transcriber or self.transcriber,
                    derivatives=derivatives,
                    config=self.config.transcription,
                ),
            )
        return EnrichmentService(
            assets=self.assets,
            scene_reader=self.scenes,
            writer=writer or self.writer,
            enrichers=enrichers,
            clock=self.clock,
            config=self.config,
        )

    def variant(
        self, *, dispatcher: Dispatcher | None = None, service: EnrichmentService | None = None
    ) -> Scheduler:
        return Scheduler(
            assets=self.assets,
            scene_reader=self.scenes,
            enrichment_reader=self.store,
            enricher=service or self.service,
            dispatcher=dispatcher or SerialDispatcher(),
        )


def make_enrichment_harness(
    *scene_sets: models.SceneSetView, dispatcher: Dispatcher | None = None
) -> EnrichmentHarness:
    config = EnrichmentConfig()
    assets = FakeAssetVersionReader()
    scenes = FakeSceneSetReader()
    store = FakeEnrichmentStore()
    writer = RecordingEnrichmentWriter(store)
    derivatives = FakeDerivativeReader()
    vision = FakeVisionModel()
    transcriber = FakeTranscriber()
    clock = FakeClock()
    description = DescriptionEnricher(
        vision=vision, derivatives=derivatives, config=config.description
    )
    transcription = TranscriptionEnricher(
        transcriber=transcriber, derivatives=derivatives, config=config.transcription
    )
    service = EnrichmentService(
        assets=assets,
        scene_reader=scenes,
        writer=writer,
        enrichers=(description, transcription),
        clock=clock,
        config=config,
    )
    scheduler = Scheduler(
        assets=assets,
        scene_reader=scenes,
        enrichment_reader=store,
        enricher=service,
        dispatcher=dispatcher or SerialDispatcher(),
    )
    harness = EnrichmentHarness(
        assets=assets,
        scenes=scenes,
        store=store,
        writer=writer,
        derivatives=derivatives,
        vision=vision,
        transcriber=transcriber,
        clock=clock,
        config=config,
        description=description,
        transcription=transcription,
        service=service,
        scheduler=scheduler,
    )
    harness.seed(*scene_sets)
    return harness
