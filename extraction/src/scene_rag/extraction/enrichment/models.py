"""Domain model of the enrichment step: read views, semantic segments, enrichment records."""

import dataclasses
import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Final, Literal

SCHEMA_VERSION: Final = 1
STEP_NAME: Final = "enrichment"
SCENE_SET_INPUT_PREFIX: Final = "scene-set:"
DERIVATIVE_INPUT_PREFIX: Final = "derivative:"


def truncate_to_millisecond(moment: datetime) -> datetime:
    """Return ``moment`` with its microseconds truncated to the millisecond.

    The stores hold BSON dates, which carry milliseconds. Every timestamp the step produces or
    compares is truncated first, so a round-tripped value equals a fresh one.
    """
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def enrichment_version(
    scene_set_version: str, scene_index: int, enricher: str, fingerprint: str, algorithm: str
) -> str:
    """Derive the data version of an enrichment record: ``hash(input versions + fingerprint)``.

    Args:
        scene_set_version: Data version of the scene set the scene belongs to.
        scene_index: Index of the scene within the set.
        enricher: Name of the enricher that produced the record.
        fingerprint: Fingerprint of that enricher.
        algorithm: ``hashlib`` algorithm name.

    Returns:
        ``<algorithm>:<hex digest>``, usable as the record's identifier.
    """
    digest = hashlib.new(
        algorithm, f"{scene_set_version}\n{scene_index}\n{enricher}\n{fingerprint}".encode()
    )
    return f"{algorithm}:{digest.hexdigest()}"


@dataclasses.dataclass(frozen=True)
class AssetVersionView:
    """What the step needs of an asset version: whether it is still live.

    Attributes:
        version: Content hash of the bytes, ``<algorithm>:<hex digest>``.
        has_sources: Whether at least one source reference still holds the version.
        retired_at: Tombstone set when the last reference was removed; ``None`` while live.
    """

    version: str
    has_sources: bool
    retired_at: datetime | None = None

    @property
    def is_live(self) -> bool:
        """Return whether a source still holds the version and it is not retired."""
        return self.has_sources and self.retired_at is None


@dataclasses.dataclass(frozen=True)
class DerivativeRefView:
    """A file in the Derivative Store, as a scene set references it.

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
class KeyframeView:
    """One rendered frame of a scene.

    Attributes:
        derivative: The image in the Derivative Store.
        frame: Index of the frame in the video stream.
        seconds: Position of the frame in the stream.
    """

    derivative: DerivativeRefView
    frame: int
    seconds: float


@dataclasses.dataclass(frozen=True)
class SceneView:
    """A scene of a scene set, as far as enrichers need it.

    Attributes:
        index: Position among the set's scenes, from zero.
        start_seconds: Start of the scene in the stream.
        end_seconds: End of the scene in the stream.
        keyframes: Rendered frames in stream order.
        audio_clip: The scene's audio, ``None`` when the asset has no audio stream or the
            scene is empty.
    """

    index: int
    start_seconds: float
    end_seconds: float
    keyframes: tuple[KeyframeView, ...] = ()
    audio_clip: DerivativeRefView | None = None

    @property
    def duration_seconds(self) -> float:
        """Return the length of the scene."""
        return self.end_seconds - self.start_seconds


@dataclasses.dataclass(frozen=True)
class SceneSetView:
    """The step's read view of a scene set written by segmentation.

    Attributes:
        version: Data version and identifier of the set.
        asset_version: Content hash of the asset the scenes were cut from.
        scenes: The scenes in order.
    """

    version: str
    asset_version: str
    scenes: tuple[SceneView, ...]


@dataclasses.dataclass(frozen=True)
class SceneSetSummary:
    """What a pass needs to derive the pending work of a scene set without loading it.

    Attributes:
        version: Data version and identifier of the set.
        asset_version: Content hash of the asset the scenes were cut from.
        scene_count: Number of scenes; their indices are ``0 .. scene_count - 1``.
    """

    version: str
    asset_version: str
    scene_count: int


@dataclasses.dataclass(frozen=True)
class ModelRef:
    """A model as the fingerprint pins it.

    Attributes:
        name: Name the model is known by to its backend (a family, possibly a floating tag).
        digest: Digest of the model's bytes, resolved by the backend; never a floating tag.
    """

    name: str
    digest: str


@dataclasses.dataclass(frozen=True)
class SemanticSegment:
    """A span of a scene with text an enricher attributed to it.

    Attributes:
        start_seconds: Start of the span in the stream.
        end_seconds: End of the span in the stream.
        text: What was seen, said or shown.
        language: Language of the text, when the enricher detects one.
        confidence: The enricher's confidence in the text, 0 to 1, when it reports one.
    """

    start_seconds: float
    end_seconds: float
    text: str
    language: str | None = None
    confidence: float | None = None


@dataclasses.dataclass(frozen=True)
class EnricherOutput:
    """What one enricher produced for one scene.

    Attributes:
        segments: Semantic segments in stream order; empty when the scene gave the enricher
            nothing to work on, or nothing to say.
        inputs: Digests of the derivatives read to produce the segments.
    """

    segments: tuple[SemanticSegment, ...] = ()
    inputs: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Lineage:
    """How an enrichment record was produced.

    Attributes:
        fingerprint: Fingerprint of the enricher that wrote the record.
        inputs: Data versions read to produce the record: the scene set and the derivatives.
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
class Enrichment:
    """One enricher's output for one scene of one scene set: the record the step writes.

    Attributes:
        version: Data version and identifier, see `enrichment_version`.
        scene_set_version: Data version of the scene set the scene belongs to.
        asset_version: Content hash of the asset, denormalised for liveness and purge queries.
        scene_index: Index of the scene within the set.
        enricher: Name of the enricher.
        classification: Data classification of the text the enricher produces.
        model: The model that produced the segments.
        segments: The semantic segments; may be empty.
        created_at: When the record was written, in UTC.
        lineage: Provenance of the record.
        schema_version: Version of the stored record's schema.
    """

    version: str
    scene_set_version: str
    asset_version: str
    scene_index: int
    enricher: str
    classification: str
    model: ModelRef
    segments: tuple[SemanticSegment, ...]
    created_at: datetime
    lineage: Lineage
    schema_version: int = SCHEMA_VERSION


type Outcome = Literal["enriched", "duplicate", "dropped", "deferred", "failed"]
"""What became of one scene set in a pass; the field names of `EnrichmentReport`."""


@dataclasses.dataclass(frozen=True)
class EnrichmentReport:
    """Counters of one enrichment pass, per scene set.

    Attributes:
        enriched: Scene sets for which at least one record was written.
        duplicate: Scene sets whose pending records another worker wrote first; nothing changed.
        dropped: Scene sets whose asset version was retired, or which vanished, while being
            enriched; nothing further written.
        deferred: Scene sets a derivative of which could not be read as segmentation filed it;
            left pending for a later pass.
        failed: Scene sets that raised; the pass continued past them.
    """

    enriched: int = 0
    duplicate: int = 0
    dropped: int = 0
    deferred: int = 0
    failed: int = 0

    @classmethod
    def counting(cls, outcomes: Iterable[Outcome]) -> EnrichmentReport:
        """Tally the outcomes of a pass, in whatever order they arrive."""
        counts: Counter[str] = Counter(outcomes)
        return cls(**{field.name: counts[field.name] for field in dataclasses.fields(cls)})

    def __str__(self) -> str:
        """Render the counters as ``name=value`` pairs for logs."""
        return " ".join(
            f"{field.name}={getattr(self, field.name)}" for field in dataclasses.fields(self)
        )
