"""Fakes and builders for the tests of the segmentation step."""

import dataclasses
import hashlib
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.dispatch import SerialDispatcher
from scene_rag.extraction.segmentation.protocols import (
    Analysis,
    AudioExtractionError,
    Dispatcher,
    KeyframeRenderer,
    MediaUnavailableError,
    RenderedFrame,
    SceneStoreWriter,
    SegmentExtractor,
    VideoAnalysisError,
)
from scene_rag.extraction.segmentation.scheduler import Scheduler
from scene_rag.extraction.segmentation.service import SegmentationService

T0 = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)
SEGMENT_BOUNDS: tuple[tuple[float, float], ...] = ((0.0, 4.0), (4.0, 8.0), (8.0, 12.5))
FPS = 25.0


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
    asset_id: str = "asset-0000000000000001",
    uri: str = "file:///media/clip.mp4",
    size_bytes: int = 4,
    modified_at: datetime = T0 - timedelta(minutes=1),
) -> models.SourceView:
    return models.SourceView(
        source_id="local",
        asset_id=asset_id,
        uri=uri,
        size_bytes=size_bytes,
        modified_at=modified_at,
    )


def make_view(
    version: str = sha256_version(b"abcd"),
    *sources: models.SourceView,
    retired_at: datetime | None = None,
) -> models.AssetVersionView:
    held = sources or (make_source(),)
    return models.AssetVersionView(
        version=version,
        media_type="video/mp4",
        size_bytes=held[0].size_bytes,
        sources=tuple(held),
        retired_at=retired_at,
    )


def make_media(duration: float = 12.5, fps: float = FPS) -> models.MediaInfo:
    return models.MediaInfo(
        duration_seconds=duration,
        frame_rate=fps,
        width=1920,
        height=1080,
        frame_count=round(duration * fps),
    )


def make_segments(
    bounds: Sequence[tuple[float, float]] = SEGMENT_BOUNDS, fps: float = FPS
) -> tuple[models.Segment, ...]:
    return tuple(
        models.Segment(
            index=index,
            start_seconds=start,
            end_seconds=end,
            start_frame=round(start * fps),
            end_frame=round(end * fps),
        )
        for index, (start, end) in enumerate(bounds)
    )


def make_analysis(
    bounds: Sequence[tuple[float, float]] = SEGMENT_BOUNDS, duration: float = 12.5
) -> Analysis:
    return Analysis(media=make_media(duration), segments=make_segments(bounds))


def make_derivative(data: bytes = b"jpeg", media_type: str = "image/jpeg") -> models.DerivativeRef:
    digest = hashlib.sha256(data).hexdigest()
    return models.DerivativeRef(
        digest=f"sha256:{digest}",
        uri=f"file:///derivatives/sha256/{digest[:2]}/{digest}.jpg",
        media_type=media_type,
        size_bytes=len(data),
    )


def make_scene_set(
    asset_version: str = sha256_version(b"abcd"),
    fingerprint: str = "0123456789ab",
    scenes: tuple[models.Scene, ...] | None = None,
    created_at: datetime = T0,
) -> models.SceneSet:
    version = models.scene_set_version(asset_version, fingerprint, "sha256")
    if scenes is None:
        keyframe = models.Keyframe(derivative=make_derivative(), frame=12, seconds=0.48)
        scenes = tuple(
            models.Scene(
                **dataclasses.asdict(segment),
                keyframes=(keyframe,),
                audio_clip=make_derivative(b"wav", "audio/wav") if segment.index == 0 else None,
            )
            for segment in make_segments()
        )
    return models.SceneSet(
        version=version,
        asset_version=asset_version,
        media=make_media(),
        scenes=scenes,
        created_at=created_at,
        lineage=models.Lineage(
            fingerprint=fingerprint,
            inputs=(models.ASSET_VERSION_INPUT_PREFIX + asset_version,),
            version=version,
        ),
    )


class FakeAssetVersionReader:
    """AssetVersionReader over a mutable dictionary of views."""

    def __init__(self, *records: models.AssetVersionView) -> None:
        self.records: dict[str, models.AssetVersionView] = {r.version: r for r in records}

    def get(self, version: str) -> models.AssetVersionView | None:
        return self.records.get(version)

    def list_live(self) -> Sequence[models.AssetVersionView]:
        return [record for _, record in sorted(self.records.items()) if record.is_live]

    def retire(self, version: str, at: datetime = T0) -> None:
        self.records[version] = dataclasses.replace(
            self.records[version], sources=(), retired_at=at
        )


class FakeSceneStore:
    """Scene Store in a dictionary, with the command semantics of the MongoDB backend.

    `tests/segmentation/test_scene_store.py` runs the same cases against both, so the fake
    stays faithful.
    """

    def __init__(self) -> None:
        self._records: dict[str, models.SceneSet] = {}

    def get(self, version: str) -> models.SceneSet | None:
        return self._records.get(version)

    def snapshot(self) -> dict[str, models.SceneSet]:
        """Return every stored scene set, for whole-state assertions."""
        return dict(self._records)

    def list_asset_versions(self, fingerprint: str) -> frozenset[str]:
        return frozenset(
            record.asset_version
            for record in self._records.values()
            if record.lineage.fingerprint == fingerprint
        )

    def put(self, scene_set: models.SceneSet) -> bool:
        if scene_set.version in self._records:
            return False
        self._records[scene_set.version] = scene_set
        return True


class RecordingSceneWriter:
    """SceneStoreWriter that logs every command, delegating to a real writer."""

    def __init__(self, inner: SceneStoreWriter, fail: bool = False) -> None:
        self._inner = inner
        self._fail = fail
        self.commands: list[tuple[str, str]] = []

    def put(self, scene_set: models.SceneSet) -> bool:
        self.commands.append(("put", scene_set.asset_version))
        if self._fail:
            raise RuntimeError("injected failure in put")
        return self._inner.put(scene_set)


class FakeDerivativeStore:
    """Content-addressed store in memory: consumes the file, keeps its bytes by digest."""

    def __init__(self, fail: bool = False) -> None:
        self.blobs: dict[str, bytes] = {}
        self.puts: list[tuple[str, str]] = []
        self.paths: list[Path] = []
        self._fail = fail

    def put(self, path: Path, media_type: str) -> models.DerivativeRef:
        if self._fail:
            raise OSError("derivative store full")
        self.paths.append(path)
        data = path.read_bytes()
        path.unlink()
        reference = make_derivative(data, media_type)
        self.blobs[reference.digest] = data
        self.puts.append((reference.digest, media_type))
        return reference


class FakeLocator:
    """MediaLocator returning a fixed path, or refusing every version."""

    def __init__(self, path: Path = Path("/media/clip.mp4"), available: bool = True) -> None:
        self.path = path
        self.available = available
        self.located: list[str] = []

    def locate(self, record: models.AssetVersionView) -> Path:
        self.located.append(record.version)
        if not self.available:
            raise MediaUnavailableError(f"no source of {record.version} is readable as recorded")
        return self.path


class FakeSegmentExtractor:
    """SegmentExtractor returning a canned analysis, or raising."""

    def __init__(self, analysis: Analysis | None = None, fail: bool = False) -> None:
        self.analysis = make_analysis() if analysis is None else analysis
        self._fail = fail
        self.calls: list[Path] = []

    def extract(self, path: Path) -> Analysis:
        self.calls.append(path)
        if self._fail:
            raise VideoAnalysisError(f"could not open video {path}")
        return self.analysis


class FakeKeyframeRenderer:
    """KeyframeRenderer writing dummy JPEG bytes per segment.

    `per_segment` frames are rendered at the segment start plus `frame_step` per pick; with
    `duplicate_frames` every pick lands on the same frame, as it does on a very short scene,
    and the renderer reports the same frame twice: the service must file it once.
    """

    def __init__(
        self,
        per_segment: int = 2,
        frame_step: int = 10,
        duplicate_frames: bool = False,
        fail: bool = False,
    ) -> None:
        self.per_segment = per_segment
        self.frame_step = 0 if duplicate_frames else frame_step
        self.duplicate_frames = duplicate_frames
        self._fail = fail

    def render(
        self, path: Path, segments: Sequence[models.Segment], output_dir: Path
    ) -> Mapping[int, Sequence[RenderedFrame]]:
        if self._fail:
            raise VideoAnalysisError(f"could not open video {path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered: dict[int, list[RenderedFrame]] = {}
        for segment in segments:
            frames: list[RenderedFrame] = []
            for pick in range(self.per_segment):
                frame = segment.start_frame + pick * self.frame_step
                file = output_dir / f"scene_{segment.index + 1:03d}_frame_{frame}_{pick}.jpg"
                file.write_bytes(f"jpeg:{segment.index}:{frame}".encode())
                frames.append(RenderedFrame(path=file, frame=frame, seconds=frame / FPS))
            rendered[segment.index] = frames
        return rendered


class FakeAudioClipper:
    """AudioClipper writing dummy WAV bytes per segment; `silent` means no audio stream."""

    def __init__(self, silent: bool = False, fail: bool = False) -> None:
        self.silent = silent
        self._fail = fail

    def clip(
        self, path: Path, segments: Sequence[models.Segment], output_dir: Path
    ) -> Mapping[int, Path]:
        if self._fail:
            raise AudioExtractionError(f"ffmpeg failed probing {path}")
        if self.silent:
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        clips: dict[int, Path] = {}
        for segment in segments:
            if segment.duration_seconds <= 0:
                continue
            file = output_dir / f"scene_{segment.index + 1:03d}.wav"
            file.write_bytes(f"wav:{segment.index}".encode())
            clips[segment.index] = file
        return clips


class FakeSegmenter:
    """VersionSegmenter answering per version from a script: an outcome or an exception.

    Versions absent from the script are ``"segmented"``. Calls are recorded in order.
    """

    def __init__(
        self,
        script: Mapping[str, Literal["segmented", "duplicate"] | Exception] | None = None,
        fingerprint: str = "0123456789ab",
    ) -> None:
        self.script = dict(script or {})
        self.fingerprint = fingerprint
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def segment(self, record: models.AssetVersionView) -> Literal["segmented", "duplicate"]:
        with self._lock:
            self.calls.append(record.version)
        answer = self.script.get(record.version, "segmented")
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


class ConcurrencyGauge:
    """Counts calls in flight at once; wrap a call in `enter` to measure a dispatcher."""

    def __init__(self) -> None:
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)

    def leave(self) -> None:
        with self._lock:
            self._active -= 1


@dataclasses.dataclass
class SegmentationHarness:
    """A segmentation pass wired to fakes, exposing them for assertions.

    `service` is the per-version unit of work, `scheduler` the pass over it; both share the
    fakes, and `variant` rebuilds the pair with some collaborators replaced.
    """

    assets: FakeAssetVersionReader
    scene_store: FakeSceneStore
    writer: RecordingSceneWriter
    derivatives: FakeDerivativeStore
    locator: FakeLocator
    extractor: FakeSegmentExtractor
    keyframes: FakeKeyframeRenderer
    audio: FakeAudioClipper
    clock: FakeClock
    service: SegmentationService
    scheduler: Scheduler

    def variant_service(
        self,
        *,
        writer: SceneStoreWriter | None = None,
        derivatives: FakeDerivativeStore | None = None,
        locator: FakeLocator | None = None,
        extractor: SegmentExtractor | None = None,
        keyframes: KeyframeRenderer | None = None,
        audio: FakeAudioClipper | None = None,
        config: SegmentationConfig | None = None,
        workspace_root: Path | None = None,
    ) -> SegmentationService:
        """Build a second service over the same fakes with some collaborators replaced."""
        return SegmentationService(
            assets=self.assets,
            scene_writer=writer or self.writer,
            derivatives=derivatives or self.derivatives,
            locator=locator or self.locator,
            extractor=extractor or self.extractor,
            keyframes=keyframes or self.keyframes,
            audio=audio or self.audio,
            clock=self.clock,
            config=config or SegmentationConfig(),
            workspace_root=workspace_root,
        )

    def variant(
        self,
        *,
        writer: SceneStoreWriter | None = None,
        derivatives: FakeDerivativeStore | None = None,
        locator: FakeLocator | None = None,
        extractor: SegmentExtractor | None = None,
        keyframes: KeyframeRenderer | None = None,
        audio: FakeAudioClipper | None = None,
        config: SegmentationConfig | None = None,
        workspace_root: Path | None = None,
        dispatcher: Dispatcher | None = None,
    ) -> Scheduler:
        """Build a second pass over the same fakes with some collaborators replaced."""
        service = self.variant_service(
            writer=writer,
            derivatives=derivatives,
            locator=locator,
            extractor=extractor,
            keyframes=keyframes,
            audio=audio,
            config=config,
            workspace_root=workspace_root,
        )
        return Scheduler(
            assets=self.assets,
            scene_reader=self.scene_store,
            segmenter=service,
            dispatcher=dispatcher or SerialDispatcher(),
        )


def make_segmentation_harness(
    *records: models.AssetVersionView, dispatcher: Dispatcher | None = None
) -> SegmentationHarness:
    """Wire a pass to fresh fakes over the given registry records; serial unless told."""
    assets = FakeAssetVersionReader(*records)
    scene_store = FakeSceneStore()
    writer = RecordingSceneWriter(scene_store)
    derivatives = FakeDerivativeStore()
    locator = FakeLocator()
    extractor = FakeSegmentExtractor()
    keyframes = FakeKeyframeRenderer()
    audio = FakeAudioClipper()
    clock = FakeClock()
    service = SegmentationService(
        assets=assets,
        scene_writer=writer,
        derivatives=derivatives,
        locator=locator,
        extractor=extractor,
        keyframes=keyframes,
        audio=audio,
        clock=clock,
        config=SegmentationConfig(),
    )
    scheduler = Scheduler(
        assets=assets,
        scene_reader=scene_store,
        segmenter=service,
        dispatcher=dispatcher or SerialDispatcher(),
    )
    return SegmentationHarness(
        assets,
        scene_store,
        writer,
        derivatives,
        locator,
        extractor,
        keyframes,
        audio,
        clock,
        service,
        scheduler,
    )
