"""Domain model of the scene-api role: read views of the stores, and the records it publishes."""

import dataclasses
import hashlib
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Final

SCHEMA_VERSION: Final = 1
ROLE_NAME: Final = "scene-api"
DEFAULT_HASH_ALGORITHM: Final = "sha256"

_FINGERPRINT: Final = re.compile(r"^[0-9a-f]{12}$")
_ENRICHER: Final = re.compile(r"^[a-z][a-z0-9_]*$")


def scene_record_version(
    scene_set_version: str,
    scene_index: int,
    enrichment_versions: Iterable[str],
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> str:
    """Derive the data version of a scene record from the versions it was joined from.

    The enrichment versions are sorted first, so the result does not depend on the order the
    store returned them in; it changes exactly when the set of enrichments included changes.

    Args:
        scene_set_version: Data version of the scene set the scene belongs to.
        scene_index: Index of the scene within the set.
        enrichment_versions: Data versions of the enrichment records included.
        algorithm: ``hashlib`` algorithm name.

    Returns:
        ``<algorithm>:<hex digest>``.
    """
    lines = [scene_set_version, str(scene_index), *sorted(enrichment_versions)]
    digest = hashlib.new(algorithm, "\n".join(lines).encode())
    return f"{algorithm}:{digest.hexdigest()}"


@dataclasses.dataclass(frozen=True)
class SourceView:
    """A source system's reference to an asset version, as far as consumers may see it.

    Attributes:
        source_id: Identifier of the connector that found the item.
        asset_id: Stable identifier of the item within its source.
        filename: Display name of the item in its source.
    """

    source_id: str
    asset_id: str
    filename: str


@dataclasses.dataclass(frozen=True)
class AssetVersionView:
    """The role's view of an asset version written by ingestion.

    Attributes:
        version: Content hash of the bytes, ``<algorithm>:<hex digest>``.
        media_type: MIME type of the bytes.
        size_bytes: Size of the bytes.
        sources: Every source reference currently holding the version; empty once retired.
        entitlements: Union of the sources' entitlements.
        retired_at: Tombstone set when the last reference was removed; ``None`` while live.
    """

    version: str
    media_type: str
    size_bytes: int
    sources: tuple[SourceView, ...]
    entitlements: frozenset[str]
    retired_at: datetime | None = None

    @property
    def is_live(self) -> bool:
        """Return whether a source still holds the version and it is not retired."""
        return bool(self.sources) and self.retired_at is None


@dataclasses.dataclass(frozen=True)
class MediaView:
    """Properties of the asset's video stream, as segmentation measured them.

    Attributes:
        duration_seconds: Length of the stream.
        frame_rate: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frame_count: Frames in the stream.
    """

    duration_seconds: float
    frame_rate: float
    width: int
    height: int
    frame_count: int


@dataclasses.dataclass(frozen=True)
class DerivativeRefView:
    """A file in the Derivative Store, as a scene set references it.

    Attributes:
        digest: Content hash of the file, ``<algorithm>:<hex digest>``.
        uri: Where the store filed it; internal, never published.
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
    """A scene of a scene set.

    Attributes:
        index: Position among the set's scenes, from zero.
        start_seconds: Start of the scene in the stream.
        end_seconds: End of the scene in the stream.
        start_frame: First frame of the scene.
        end_frame: Frame after the last frame of the scene.
        keyframes: Rendered frames in stream order.
        audio_clip: The scene's audio; ``None`` when the asset has no audio stream or the
            scene is empty.
    """

    index: int
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    keyframes: tuple[KeyframeView, ...] = ()
    audio_clip: DerivativeRefView | None = None

    @property
    def derivatives(self) -> tuple[DerivativeRefView, ...]:
        """Return every derivative the scene references, keyframes first."""
        clips = () if self.audio_clip is None else (self.audio_clip,)
        return tuple(keyframe.derivative for keyframe in self.keyframes) + clips


@dataclasses.dataclass(frozen=True)
class SceneSetView:
    """The role's view of a scene set written by segmentation.

    Attributes:
        version: Data version and identifier of the set.
        asset_version: Content hash of the asset the scenes were cut from.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut it.
        media: Properties of the video stream.
        scenes: The scenes in order.
        created_at: When the set was written, in UTC.
    """

    version: str
    asset_version: str
    segmentation_fingerprint: str
    media: MediaView
    scenes: tuple[SceneView, ...]
    created_at: datetime

    @property
    def summary(self) -> SceneSetSummaryView:
        """Return the set's summary, as a listing would project it."""
        return SceneSetSummaryView(
            version=self.version,
            asset_version=self.asset_version,
            segmentation_fingerprint=self.segmentation_fingerprint,
            scene_count=len(self.scenes),
            created_at=self.created_at,
        )


@dataclasses.dataclass(frozen=True)
class SceneSetSummaryView:
    """What a listing projects of a scene set, without loading its scenes.

    Attributes:
        version: Data version and identifier of the set.
        asset_version: Content hash of the asset the scenes were cut from.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut it.
        scene_count: Number of scenes; their indices are ``0 .. scene_count - 1``.
        created_at: When the set was written, in UTC.
    """

    version: str
    asset_version: str
    segmentation_fingerprint: str
    scene_count: int
    created_at: datetime


@dataclasses.dataclass(frozen=True)
class ModelRef:
    """A model as an enrichment record pins it.

    Attributes:
        name: Name the model is known by to its backend.
        digest: Digest of the model's bytes.
    """

    name: str
    digest: str


@dataclasses.dataclass(frozen=True)
class SemanticSegmentView:
    """A span of a scene with text an enricher attributed to it.

    Attributes:
        start_seconds: Start of the span in the stream.
        end_seconds: End of the span in the stream.
        text: What was seen, said or shown.
        language: Language of the text, when the enricher detected one.
        confidence: The enricher's confidence in the text, 0 to 1, when it reported one.
    """

    start_seconds: float
    end_seconds: float
    text: str
    language: str | None = None
    confidence: float | None = None


@dataclasses.dataclass(frozen=True)
class EnrichmentView:
    """The role's view of an enrichment record written by enrichment.

    Attributes:
        version: Data version and identifier of the record.
        scene_set_version: Data version of the scene set the scene belongs to.
        scene_index: Index of the scene within the set.
        enricher: Name of the enricher.
        fingerprint: Fingerprint of the enricher that wrote the record.
        classification: Data classification of the text.
        model: The model that produced the segments.
        segments: The semantic segments; may be empty.
    """

    version: str
    scene_set_version: str
    scene_index: int
    enricher: str
    fingerprint: str
    classification: str
    model: ModelRef
    segments: tuple[SemanticSegmentView, ...] = ()

    @property
    def pin(self) -> EnrichmentPin:
        """Return the (enricher, fingerprint) pair the record was written under."""
        return EnrichmentPin(self.enricher, self.fingerprint)


@dataclasses.dataclass(frozen=True)
class Coverage:
    """How many scenes of a scene set have a record under one enricher fingerprint.

    Attributes:
        scene_set_version: Data version of the scene set.
        enricher: Name of the enricher.
        fingerprint: Fingerprint of the enricher.
        scene_count: Distinct scenes with a record; at least one.
    """

    scene_set_version: str
    enricher: str
    fingerprint: str
    scene_count: int


@dataclasses.dataclass(frozen=True, order=True)
class EnrichmentPin:
    """An enricher at one fingerprint, as a consumer pins it in a request.

    Attributes:
        enricher: Name of the enricher.
        fingerprint: Fingerprint the records must have been written under.
    """

    enricher: str
    fingerprint: str

    @classmethod
    def parse(cls, text: str) -> EnrichmentPin:
        """Parse ``<enricher>:<fingerprint>``.

        Raises:
            ValueError: When the text is not an enricher name, a colon and twelve hex digits.
        """
        enricher, colon, fingerprint = text.partition(":")
        if not colon or not _ENRICHER.match(enricher) or not _FINGERPRINT.match(fingerprint):
            raise ValueError(
                f"enrichment pin must be <enricher>:<fingerprint> (twelve hex digits), not {text!r}"
            )
        return cls(enricher, fingerprint)

    def __str__(self) -> str:
        """Render the pin as it is written in a request."""
        return f"{self.enricher}:{self.fingerprint}"


@dataclasses.dataclass(frozen=True)
class EnrichmentCoverage:
    """Coverage of one scene set by one enricher fingerprint, as a summary publishes it.

    Attributes:
        enricher: Name of the enricher.
        fingerprint: Fingerprint of the enricher.
        scene_count: Scenes of the set with a record under the fingerprint.
        complete: Whether every scene of the set has one.
    """

    enricher: str
    fingerprint: str
    scene_count: int
    complete: bool


@dataclasses.dataclass(frozen=True)
class SceneSetSummary:
    """A scene set as the role lists it.

    Attributes:
        version: Data version and identifier of the set.
        asset_version: Content hash of the asset the scenes were cut from.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut it.
        scene_count: Number of scenes.
        created_at: When the set was written, in UTC.
        live: Whether a source still holds the asset version.
        enrichments: Coverage per (enricher, fingerprint), sorted.
    """

    version: str
    asset_version: str
    segmentation_fingerprint: str
    scene_count: int
    created_at: datetime
    live: bool
    enrichments: tuple[EnrichmentCoverage, ...]


@dataclasses.dataclass(frozen=True)
class SceneRecord:
    """A scene with its enrichments: the record the role publishes.

    Attributes:
        version: Data version, see `scene_record_version`.
        scene_set_version: Data version of the scene set the scene belongs to.
        asset_version: Content hash of the asset the scene was cut from.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut it.
        live: Whether a source still holds the asset version.
        asset: The asset version with its entitlements and source ids.
        media: Properties of the video stream.
        scene: The scene with its derivative references.
        enrichments: The enrichments included, sorted by (enricher, fingerprint).
        schema_version: Version of the published record's schema.
    """

    version: str
    scene_set_version: str
    asset_version: str
    segmentation_fingerprint: str
    live: bool
    asset: AssetVersionView
    media: MediaView
    scene: SceneView
    enrichments: tuple[EnrichmentView, ...]
    schema_version: int = SCHEMA_VERSION


@dataclasses.dataclass(frozen=True)
class SceneSetQuery:
    """What a consumer asks of the scene set listing.

    Attributes:
        asset_version: Only sets cut from this asset version, when given.
        segmentation_fingerprint: Only sets cut under this fingerprint, when given.
        pins: Only sets complete under every pin; no pins means no completeness filter.
        after: Cursor of the previous page: the last scene set version scanned.
        limit: Most sets to scan for the page; the service's page size when ``None``.
    """

    asset_version: str | None = None
    segmentation_fingerprint: str | None = None
    pins: tuple[EnrichmentPin, ...] = ()
    after: str | None = None
    limit: int | None = None


@dataclasses.dataclass(frozen=True)
class SceneQuery:
    """What a consumer asks of the scene record listing of one scene set.

    Attributes:
        pins: Only enrichments written under these pins; no pins means every enrichment.
        after: Cursor of the previous page: the index of the last scene returned.
        limit: Most scenes on the page; the service's page size when ``None``.
    """

    pins: tuple[EnrichmentPin, ...] = ()
    after: int | None = None
    limit: int | None = None


@dataclasses.dataclass(frozen=True)
class Page[T]:
    """One page of a listing.

    Attributes:
        items: The items of the page; possibly fewer than asked, even none, with more to come.
        next: Cursor of the following page; ``None`` on the last one.
    """

    items: tuple[T, ...]
    next: str | None


@dataclasses.dataclass(frozen=True)
class DerivativeContent:
    """The bytes of a derivative with the reference they were verified against.

    Attributes:
        reference: The reference, naming the digest and media type.
        data: The bytes.
    """

    reference: DerivativeRefView
    data: bytes
