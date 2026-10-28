"""Domain model of the chunking step: read views of scene-api, drafts, chunks, chunk sets."""

import dataclasses
import hashlib
from collections.abc import Iterable
from datetime import datetime
from typing import Final

from scene_rag.retrieval.chunking.config import (
    ENRICHER_PATTERN,
    FINGERPRINT_PATTERN,
    UpstreamConfig,
)

SCHEMA_VERSION: Final = 1
STEP_NAME: Final = "chunking"
SCENE_RECORD_INPUT_PREFIX: Final = "scene-record:"
SCENE_RECORD_SCHEMA_VERSION: Final = 1
"""The published scene record schema this step decodes; see `scene_api_client`."""


def truncate_to_millisecond(moment: datetime) -> datetime:
    """Return ``moment`` with its microseconds truncated to the millisecond.

    The stores hold BSON dates, which carry milliseconds. Every timestamp the step produces or
    compares is truncated first, so a round-tripped value equals a fresh one.
    """
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def chunk_version(
    scene_record_versions: Iterable[str], fingerprint: str, chunk_index: int, algorithm: str
) -> str:
    """Derive the data version of a chunk: ``hash(input versions + fingerprint)``.

    The record versions are sorted first, so the result does not depend on the order a chunker
    lists the scenes it covers.

    Args:
        scene_record_versions: Data versions of the scene records the chunk was built from.
        fingerprint: Fingerprint of the step.
        chunk_index: Position of the chunk among those of its scene set.
        algorithm: ``hashlib`` algorithm name.

    Returns:
        ``<algorithm>:<hex digest>``, usable as the record's identifier.
    """
    lines = [*sorted(scene_record_versions), fingerprint, str(chunk_index)]
    digest = hashlib.new(algorithm, "\n".join(lines).encode())
    return f"{algorithm}:{digest.hexdigest()}"


def chunk_set_version(scene_set_version: str, fingerprint: str, algorithm: str) -> str:
    """Derive the data version of a chunk set: ``hash(scene set version + fingerprint)``."""
    digest = hashlib.new(algorithm, f"{scene_set_version}\n{fingerprint}".encode())
    return f"{algorithm}:{digest.hexdigest()}"


@dataclasses.dataclass(frozen=True, order=True)
class EnrichmentPin:
    """An enricher at one fingerprint, as scene-api takes it in a request.

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
        if (
            not colon
            or not ENRICHER_PATTERN.match(enricher)
            or not FINGERPRINT_PATTERN.match(fingerprint)
        ):
            raise ValueError(
                f"enrichment pin must be <enricher>:<fingerprint> (twelve hex digits), not {text!r}"
            )
        return cls(enricher, fingerprint)

    def __str__(self) -> str:
        """Render the pin as it is written in a request."""
        return f"{self.enricher}:{self.fingerprint}"


@dataclasses.dataclass(frozen=True)
class SceneSetQuery:
    """What the step asks of the scene set listing.

    Attributes:
        segmentation_fingerprint: Only sets cut under this fingerprint, when given.
        pins: Only sets complete under every pin; no pins means every set.
    """

    segmentation_fingerprint: str | None = None
    pins: tuple[EnrichmentPin, ...] = ()


@dataclasses.dataclass(frozen=True)
class SourceView:
    """A source system's reference to an asset version, as scene-api publishes it.

    Attributes:
        source_id: Identifier of the connector that found the item.
        asset_id: Stable identifier of the item within its source.
        filename: Display name of the item in its source.
    """

    source_id: str
    asset_id: str
    filename: str


@dataclasses.dataclass(frozen=True)
class AssetView:
    """What a scene record says of its asset version.

    Attributes:
        media_type: MIME type of the bytes.
        size_bytes: Size of the bytes.
        entitlements: Union of the entitlements of every source holding the version.
        sources: Source references currently holding the version.
    """

    media_type: str
    size_bytes: int
    entitlements: frozenset[str]
    sources: tuple[SourceView, ...]


@dataclasses.dataclass(frozen=True)
class AssetVersionView:
    """An asset version as scene-api answers a direct fetch: the step needs its liveness.

    Attributes:
        version: Content hash of the bytes.
        live: Whether a source still holds the version.
    """

    version: str
    live: bool


@dataclasses.dataclass(frozen=True)
class MediaView:
    """Properties of the asset's video stream.

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
    """A keyframe or audio clip, addressed by content and served by scene-api.

    Attributes:
        digest: Content hash of the file.
        media_type: MIME type of the file.
        size_bytes: Size of the file.
        url: Path of the bytes on the scene-api, relative to its base URL.
    """

    digest: str
    media_type: str
    size_bytes: int
    url: str


@dataclasses.dataclass(frozen=True)
class KeyframeView:
    """One rendered frame of a scene.

    Attributes:
        derivative: The image.
        frame: Index of the frame in the video stream.
        seconds: Position of the frame in the stream.
    """

    derivative: DerivativeRefView
    frame: int
    seconds: float


@dataclasses.dataclass(frozen=True)
class SceneView:
    """A scene with its derivative references.

    Attributes:
        index: Position among the set's scenes, from zero.
        start_seconds: Start of the scene in the stream.
        end_seconds: End of the scene in the stream.
        start_frame: First frame of the scene.
        end_frame: Frame after the last frame of the scene.
        keyframes: Rendered frames in stream order.
        audio_clip: The scene's audio; ``None`` when there is none.
    """

    index: int
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    keyframes: tuple[KeyframeView, ...] = ()
    audio_clip: DerivativeRefView | None = None


@dataclasses.dataclass(frozen=True)
class SegmentView:
    """A span of the scene with text an enricher attributed to it.

    Attributes:
        start_seconds: Start of the span in the stream.
        end_seconds: End of the span in the stream.
        text: What was seen, said or shown.
        language: Language of the text, when detected.
        confidence: The enricher's confidence in the text, when reported.
    """

    start_seconds: float
    end_seconds: float
    text: str
    language: str | None = None
    confidence: float | None = None


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
class EnrichmentView:
    """One enrichment of a scene record.

    Attributes:
        version: Data version of the enrichment record.
        enricher: Name of the enricher.
        fingerprint: Fingerprint the record was written under.
        classification: Data classification of the text.
        model: The model that produced the segments.
        segments: The semantic segments; may be empty.
    """

    version: str
    enricher: str
    fingerprint: str
    classification: str
    model: ModelRef
    segments: tuple[SegmentView, ...] = ()

    @property
    def pin(self) -> EnrichmentPin:
        """Return the (enricher, fingerprint) pair the record was written under."""
        return EnrichmentPin(self.enricher, self.fingerprint)

    @property
    def ref(self) -> EnrichmentRef:
        """Return what a chunk keeps of the record."""
        return EnrichmentRef(self.enricher, self.fingerprint, self.version)


@dataclasses.dataclass(frozen=True)
class SceneRecordView:
    """A scene record as scene-api publishes it: the step's input.

    Attributes:
        version: Data version of the record.
        scene_set_version: Data version of the scene set the scene belongs to.
        asset_version: Content hash of the asset the scene was cut from.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut it.
        live: Whether a source still holds the asset version.
        asset: The asset version with its entitlements and source ids.
        media: Properties of the video stream.
        scene: The scene with its derivative references.
        enrichments: The enrichments included, sorted by (enricher, fingerprint).
    """

    version: str
    scene_set_version: str
    asset_version: str
    segmentation_fingerprint: str
    live: bool
    asset: AssetView
    media: MediaView
    scene: SceneView
    enrichments: tuple[EnrichmentView, ...]


@dataclasses.dataclass(frozen=True)
class CoverageView:
    """Coverage of a scene set by one enricher fingerprint, as a summary publishes it.

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
class SceneSetSummaryView:
    """A scene set as scene-api lists it.

    Attributes:
        version: Data version of the set.
        asset_version: Content hash of the asset the scenes were cut from.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut it.
        scene_count: Number of scenes.
        created_at: When the set was written, in UTC.
        live: Whether a source still holds the asset version.
        enrichments: Coverage per (enricher, fingerprint).
    """

    version: str
    asset_version: str
    segmentation_fingerprint: str
    scene_count: int
    created_at: datetime
    live: bool
    enrichments: tuple[CoverageView, ...] = ()


def resolve_pins(
    summary: SceneSetSummaryView, upstream: UpstreamConfig
) -> tuple[EnrichmentPin, ...] | None:
    """Decide which enrichment fingerprints a scene set is chunked under.

    For every required enricher: the configured pin when one names it, else the fingerprint
    with complete coverage in the summary, the greatest when several are complete, so the
    choice is deterministic. A set lacking complete coverage for a required enricher is not
    ready. A configured pin is honoured only when the summary shows it complete.

    Args:
        summary: The scene set as listed.
        upstream: The accepted Media outputs.

    Returns:
        One pin per required enricher, in the order required; ``None`` when the set is not
        ready to be chunked.
    """
    configured = {
        pin.enricher: pin for pin in (EnrichmentPin.parse(p) for p in upstream.enrichment_pins)
    }
    complete = {
        EnrichmentPin(item.enricher, item.fingerprint)
        for item in summary.enrichments
        if item.complete
    }
    pins: list[EnrichmentPin] = []
    for enricher in upstream.required_enrichers:
        chosen = configured.get(enricher)
        if chosen is None:
            candidates = [pin for pin in complete if pin.enricher == enricher]
            chosen = max(candidates) if candidates else None
        if chosen is None or chosen not in complete:
            return None
        pins.append(chosen)
    return tuple(pins)


@dataclasses.dataclass(frozen=True)
class EnrichmentRef:
    """What a chunk keeps of an enrichment record whose text went in.

    Attributes:
        enricher: Name of the enricher.
        fingerprint: Fingerprint the record was written under.
        version: Data version of the record.
    """

    enricher: str
    fingerprint: str
    version: str


@dataclasses.dataclass(frozen=True)
class KeyframeRef:
    """A keyframe a chunk points at, for citations and thumbnails.

    Attributes:
        digest: Content hash of the image, as served by scene-api.
        seconds: Position of the frame in the stream.
    """

    digest: str
    seconds: float


@dataclasses.dataclass(frozen=True)
class ChunkDraft:
    """What a `Chunker` produces: one retrieval unit before identity and provenance are added.

    Attributes:
        scene_indices: Scenes the chunk covers, ascending.
        start_seconds: Start of the covered span in the stream.
        end_seconds: End of the covered span in the stream.
        text: The retrieval text.
        enrichments: The enrichment records whose text went in.
    """

    scene_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    text: str
    enrichments: tuple[EnrichmentRef, ...]


@dataclasses.dataclass(frozen=True)
class Lineage:
    """How a record of the step was produced.

    Attributes:
        fingerprint: Fingerprint of the step that wrote the record.
        inputs: Data versions read: the scene records.
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
class Chunk:
    """One retrieval unit with its provenance: the record the step writes.

    Attributes:
        version: Data version and identifier, see `chunk_version`.
        scene_set_version: Data version of the scene set the chunk was cut from.
        asset_version: Content hash of the asset.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut the scenes.
        chunk_index: Position among the chunks of the scene set under this fingerprint.
        scene_indices: Scenes the chunk covers, ascending.
        start_seconds: Start of the covered span in the stream.
        end_seconds: End of the covered span in the stream.
        text: The retrieval text.
        sources: Source references holding the asset version, for citations.
        entitlements: Who may see the chunk: the asset version's entitlements.
        classifications: Distinct data classifications of the text, sorted.
        keyframes: Keyframes of the covered scenes, in stream order.
        enrichments: The enrichment records whose text went in.
        created_at: When the record was written, in UTC.
        lineage: Provenance of the record.
        schema_version: Version of the stored record's schema.
    """

    version: str
    scene_set_version: str
    asset_version: str
    segmentation_fingerprint: str
    chunk_index: int
    scene_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    text: str
    sources: tuple[SourceView, ...]
    entitlements: frozenset[str]
    classifications: tuple[str, ...]
    keyframes: tuple[KeyframeRef, ...]
    enrichments: tuple[EnrichmentRef, ...]
    created_at: datetime
    lineage: Lineage
    schema_version: int = SCHEMA_VERSION


@dataclasses.dataclass(frozen=True)
class ChunkSet:
    """One scene set chunked under one fingerprint: the completion marker the step writes last.

    Attributes:
        version: Data version and identifier, see `chunk_set_version`.
        scene_set_version: Data version of the scene set.
        asset_version: Content hash of the asset.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut the scenes.
        pins: Enrichment pins the scene records were fetched under.
        chunk_count: Chunks written; zero when no scene had text.
        created_at: When the record was written, in UTC.
        lineage: Provenance: every scene record version of the set.
        schema_version: Version of the stored record's schema.
    """

    version: str
    scene_set_version: str
    asset_version: str
    segmentation_fingerprint: str
    pins: tuple[EnrichmentPin, ...]
    chunk_count: int
    created_at: datetime
    lineage: Lineage
    schema_version: int = SCHEMA_VERSION
