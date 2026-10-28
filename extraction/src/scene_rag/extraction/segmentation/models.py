"""Domain model of the segmentation step: asset version views, scene sets, derivatives."""

import dataclasses
import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Final, Literal

SCHEMA_VERSION: Final = 1
STEP_NAME: Final = "segmentation"
ASSET_VERSION_INPUT_PREFIX: Final = "asset-version:"


def truncate_to_millisecond(moment: datetime) -> datetime:
    """Return ``moment`` with its microseconds truncated to the millisecond.

    The stores hold BSON dates, which carry milliseconds. Every timestamp the step produces or
    compares is truncated first, so a round-tripped value equals a fresh one.
    """
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def scene_set_version(asset_version: str, fingerprint: str, algorithm: str) -> str:
    """Derive the data version of a scene set: ``hash(input versions + fingerprint)``.

    Args:
        asset_version: Content hash of the asset the scene set was cut from.
        fingerprint: Fingerprint of the segmentation step that cut it.
        algorithm: ``hashlib`` algorithm name.

    Returns:
        ``<algorithm>:<hex digest>``, usable as the scene set's identifier.
    """
    digest = hashlib.new(algorithm, f"{asset_version}\n{fingerprint}".encode())
    return f"{algorithm}:{digest.hexdigest()}"


@dataclasses.dataclass(frozen=True)
class SourceView:
    """What the step needs of a source reference: where the bytes are and how they looked.

    Attributes:
        source_id: Identifier of the connector that registered the reference.
        asset_id: Stable identifier of the item within its source.
        uri: Where the bytes can be read from.
        size_bytes: Size the connector saw when it hashed the bytes.
        modified_at: Modification time the connector saw, UTC, millisecond precision.
    """

    source_id: str
    asset_id: str
    uri: str
    size_bytes: int
    modified_at: datetime


@dataclasses.dataclass(frozen=True)
class AssetVersionView:
    """The step's read view of an asset version registered by ingestion.

    Attributes:
        version: Content hash of the bytes, ``<algorithm>:<hex digest>``.
        media_type: MIME type of the bytes.
        size_bytes: Size of the bytes.
        sources: Source references currently holding the bytes.
        retired_at: Tombstone set when the last reference was removed; ``None`` while live.
    """

    version: str
    media_type: str
    size_bytes: int
    sources: tuple[SourceView, ...]
    retired_at: datetime | None = None

    @property
    def is_live(self) -> bool:
        """Return whether at least one source still holds the version and it is not retired."""
        return bool(self.sources) and self.retired_at is None


@dataclasses.dataclass(frozen=True)
class MediaInfo:
    """Properties of the video stream the scenes were cut from.

    Attributes:
        duration_seconds: Length of the stream.
        frame_rate: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frame_count: Number of frames in the stream.
    """

    duration_seconds: float
    frame_rate: float
    width: int
    height: int
    frame_count: int


@dataclasses.dataclass(frozen=True)
class Segment:
    """A time span of the asset; the frame span is ``[start_frame, end_frame)``.

    Attributes:
        index: Position among the asset's segments, from zero.
        start_seconds: Start of the span.
        end_seconds: End of the span.
        start_frame: First frame of the span.
        end_frame: Frame after the last frame of the span.
    """

    index: int
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int

    @property
    def duration_seconds(self) -> float:
        """Return the length of the span."""
        return self.end_seconds - self.start_seconds


@dataclasses.dataclass(frozen=True)
class DerivativeRef:
    """A file in the Derivative Store, addressed by its content.

    Attributes:
        digest: Content hash of the file, ``<algorithm>:<hex digest>``.
        uri: Where the store filed it.
        media_type: MIME type of the file.
        size_bytes: Size of the file.
    """

    digest: str
    uri: str
    media_type: str
    size_bytes: int


@dataclasses.dataclass(frozen=True)
class Keyframe:
    """One rendered frame of a scene.

    Attributes:
        derivative: The image in the Derivative Store.
        frame: Index of the frame in the video stream.
        seconds: Position of the frame in the stream.
    """

    derivative: DerivativeRef
    frame: int
    seconds: float


@dataclasses.dataclass(frozen=True)
class Scene(Segment):
    """A segment with what the step rendered from it.

    Attributes:
        keyframes: Rendered frames in stream order.
        audio_clip: The scene's audio in the Derivative Store, ``None`` when the asset has no
            audio stream or the scene is empty.
    """

    keyframes: tuple[Keyframe, ...] = ()
    audio_clip: DerivativeRef | None = None


@dataclasses.dataclass(frozen=True)
class Lineage:
    """How a scene set was produced.

    Attributes:
        fingerprint: Fingerprint of the step that wrote the record.
        inputs: Data versions read to produce the record: the asset version.
        version: Data version of the record, ``hash(inputs + fingerprint)``.
        step: Name of the producing step.
        image: Digest of the service image, when known.
    """

    fingerprint: str
    inputs: tuple[str, ...]
    version: str
    step: str = STEP_NAME
    image: str | None = None


@dataclasses.dataclass(frozen=True)
class SceneSet:
    """One asset version's scenes from one segmentation run.

    Attributes:
        version: Data version and identifier, see `scene_set_version`.
        asset_version: Content hash of the asset the scenes were cut from.
        media: Properties of the video stream.
        scenes: The scenes in order; they tile the stream.
        created_at: When the set was written, in UTC.
        lineage: Provenance of the record.
        schema_version: Version of the stored record's schema.
    """

    version: str
    asset_version: str
    media: MediaInfo
    scenes: tuple[Scene, ...]
    created_at: datetime
    lineage: Lineage
    schema_version: int = SCHEMA_VERSION


type Outcome = Literal["segmented", "duplicate", "dropped", "deferred", "failed"]
"""What became of one version in a pass; the field names of `SegmentationReport`."""


@dataclasses.dataclass(frozen=True)
class SegmentationReport:
    """Counters of one segmentation pass.

    Attributes:
        segmented: Scene sets written.
        duplicate: Versions whose scene set another worker wrote first; nothing changed.
        dropped: Versions retired while they were being segmented; nothing written.
        deferred: Versions whose bytes could not be read as ingestion recorded them; left
            pending for a later pass, after ingestion has reconciled the source.
        failed: Versions that raised; the pass continued past them.
    """

    segmented: int = 0
    duplicate: int = 0
    dropped: int = 0
    deferred: int = 0
    failed: int = 0

    @classmethod
    def counting(cls, outcomes: Iterable[Outcome]) -> SegmentationReport:
        """Tally the outcomes of a pass, in whatever order they arrive."""
        counts: Counter[str] = Counter(outcomes)
        return cls(**{field.name: counts[field.name] for field in dataclasses.fields(cls)})

    def __str__(self) -> str:
        """Render the counters as ``name=value`` pairs for logs."""
        return " ".join(
            f"{field.name}={getattr(self, field.name)}" for field in dataclasses.fields(self)
        )
