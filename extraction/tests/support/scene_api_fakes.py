"""Fakes and builders for the tests of the scene-api role."""

import dataclasses
import hashlib
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from scene_rag.extraction.scene_api import models
from scene_rag.extraction.scene_api.protocols import (
    DerivativeUnavailableError,
    InvalidQueryError,
    NotFoundError,
)
from scene_rag.extraction.scene_api.service import SceneCatalogService

T0 = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)
SEGMENT_BOUNDS: tuple[tuple[float, float], ...] = ((0.0, 4.0), (4.0, 8.0), (8.0, 12.5))
FPS = 25.0
SEGMENTATION_FP = "0123456789ab"
DESCRIPTION_FP = "3b9e1f6a2c40"
TRANSCRIPT_FP = "e41a9c27b0d5"
PUBLIC = frozenset({"public"})


def sha256_version(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def make_source(
    asset_id: str = "asset-0000000000000001", source_id: str = "local", filename: str = "clip.mp4"
) -> models.SourceView:
    return models.SourceView(source_id=source_id, asset_id=asset_id, filename=filename)


def make_asset_view(
    version: str = sha256_version(b"abcd"),
    *sources: models.SourceView,
    entitlements: frozenset[str] = PUBLIC,
    retired_at: datetime | None = None,
) -> models.AssetVersionView:
    held = sources or (make_source(),)
    return models.AssetVersionView(
        version=version,
        media_type="video/mp4",
        size_bytes=4,
        sources=tuple(held),
        entitlements=entitlements,
        retired_at=retired_at,
    )


def retired(asset: models.AssetVersionView, at: datetime = T0) -> models.AssetVersionView:
    return dataclasses.replace(asset, sources=(), entitlements=frozenset(), retired_at=at)


def make_media(duration: float = 12.5, fps: float = FPS) -> models.MediaView:
    return models.MediaView(
        duration_seconds=duration,
        frame_rate=fps,
        width=1920,
        height=1080,
        frame_count=round(duration * fps),
    )


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


def make_scenes(
    bounds: Sequence[tuple[float, float]] = SEGMENT_BOUNDS, fps: float = FPS
) -> tuple[models.SceneView, ...]:
    keyframe = models.KeyframeView(derivative=make_derivative(), frame=12, seconds=0.48)
    return tuple(
        models.SceneView(
            index=index,
            start_seconds=start,
            end_seconds=end,
            start_frame=round(start * fps),
            end_frame=round(end * fps),
            keyframes=(keyframe,),
            audio_clip=make_derivative(b"wav", "audio/wav") if index == 0 else None,
        )
        for index, (start, end) in enumerate(bounds)
    )


def clip_of(scene_set: models.SceneSetView, index: int = 0) -> models.DerivativeRefView:
    """The audio clip of a scene the builders gave one."""
    clip = scene_set.scenes[index].audio_clip
    assert clip is not None
    return clip


def scene_set_version(asset_version: str, fingerprint: str) -> str:
    return sha256_version(f"{asset_version}\n{fingerprint}".encode())


def make_scene_set_view(
    asset_version: str = sha256_version(b"abcd"),
    fingerprint: str = SEGMENTATION_FP,
    scenes: tuple[models.SceneView, ...] | None = None,
    created_at: datetime = T0,
) -> models.SceneSetView:
    return models.SceneSetView(
        version=scene_set_version(asset_version, fingerprint),
        asset_version=asset_version,
        segmentation_fingerprint=fingerprint,
        media=make_media(),
        scenes=make_scenes() if scenes is None else scenes,
        created_at=created_at,
    )


def make_enrichment_view(
    scene_set: models.SceneSetView,
    scene_index: int,
    enricher: str = "transcript",
    fingerprint: str = TRANSCRIPT_FP,
    segments: tuple[models.SemanticSegmentView, ...] | None = None,
) -> models.EnrichmentView:
    if segments is None:
        scene = scene_set.scenes[scene_index]
        segments = (
            models.SemanticSegmentView(
                scene.start_seconds, scene.end_seconds, f"{enricher} of {scene_index}", "en", 0.9
            ),
        )
    return models.EnrichmentView(
        version=sha256_version(
            f"{scene_set.version}\n{scene_index}\n{enricher}\n{fingerprint}".encode()
        ),
        scene_set_version=scene_set.version,
        scene_index=scene_index,
        enricher=enricher,
        fingerprint=fingerprint,
        classification="internal",
        model=models.ModelRef(name=f"{enricher}-model", digest="sha256:" + "ab" * 32),
        segments=segments,
    )


def enrich_all(
    scene_set: models.SceneSetView, enricher: str, fingerprint: str
) -> tuple[models.EnrichmentView, ...]:
    return tuple(
        make_enrichment_view(scene_set, scene.index, enricher, fingerprint)
        for scene in scene_set.scenes
    )


def make_record(
    scene_set: models.SceneSetView | None = None,
    scene_index: int = 0,
    asset: models.AssetVersionView | None = None,
    enrichments: tuple[models.EnrichmentView, ...] | None = None,
) -> models.SceneRecord:
    scene_set = make_scene_set_view() if scene_set is None else scene_set
    asset = make_asset_view(scene_set.asset_version) if asset is None else asset
    if enrichments is None:
        enrichments = (make_enrichment_view(scene_set, scene_index),)
    return models.SceneRecord(
        version=models.scene_record_version(
            scene_set.version, scene_index, (item.version for item in enrichments)
        ),
        scene_set_version=scene_set.version,
        asset_version=scene_set.asset_version,
        segmentation_fingerprint=scene_set.segmentation_fingerprint,
        live=asset.is_live,
        asset=asset,
        media=scene_set.media,
        scene=scene_set.scenes[scene_index],
        enrichments=enrichments,
    )


class FakeAssetVersionReader:
    """AssetVersionReader over a mutable dictionary of views."""

    def __init__(self, *records: models.AssetVersionView) -> None:
        self.records: dict[str, models.AssetVersionView] = {r.version: r for r in records}

    def get(self, version: str) -> models.AssetVersionView | None:
        return self.records.get(version)

    def get_many(self, versions: Iterable[str]) -> Mapping[str, models.AssetVersionView]:
        return {v: self.records[v] for v in versions if v in self.records}

    def retire(self, version: str, at: datetime = T0) -> None:
        self.records[version] = retired(self.records[version], at)


class FakeSceneSetReader:
    """SceneSetReader over a dictionary, with the query semantics of the MongoDB backend.

    `tests/scene_api/test_scene_store.py` runs the same cases against both, so the fake stays
    faithful: summaries come in version order, after the cursor, filtered as asked.
    """

    def __init__(self, *scene_sets: models.SceneSetView) -> None:
        self.scene_sets: dict[str, models.SceneSetView] = {s.version: s for s in scene_sets}

    def get(self, version: str) -> models.SceneSetView | None:
        return self.scene_sets.get(version)

    def list_summaries(
        self,
        *,
        after: str | None,
        limit: int,
        asset_version: str | None,
        segmentation_fingerprint: str | None,
    ) -> Sequence[models.SceneSetSummaryView]:
        matching = (
            scene_set
            for version, scene_set in sorted(self.scene_sets.items())
            if (after is None or version > after)
            and (asset_version is None or scene_set.asset_version == asset_version)
            and (
                segmentation_fingerprint is None
                or scene_set.segmentation_fingerprint == segmentation_fingerprint
            )
        )
        return [scene_set.summary for scene_set in matching][:limit]

    def find_derivative(self, digest: str) -> models.DerivativeRefView | None:
        for _, scene_set in sorted(self.scene_sets.items()):
            for scene in scene_set.scenes:
                for reference in scene.derivatives:
                    if reference.digest == digest:
                        return reference
        return None


class FakeEnrichmentReader:
    """EnrichmentReader over a dictionary of records keyed by version."""

    def __init__(self, *records: models.EnrichmentView) -> None:
        self.records: dict[str, models.EnrichmentView] = {r.version: r for r in records}
        self.queries: list[tuple[str, frozenset[int], tuple[models.EnrichmentPin, ...]]] = []
        self._lock = threading.Lock()

    def add(self, *records: models.EnrichmentView) -> None:
        with self._lock:
            for record in records:
                self.records[record.version] = record

    def list_coverage(self, scene_set_versions: Iterable[str]) -> Sequence[models.Coverage]:
        wanted = set(scene_set_versions)
        scenes: dict[tuple[str, str, str], set[int]] = {}
        for record in self.records.values():
            if record.scene_set_version in wanted:
                key = (record.scene_set_version, record.enricher, record.fingerprint)
                scenes.setdefault(key, set()).add(record.scene_index)
        return [
            models.Coverage(version, enricher, fingerprint, len(indices))
            for (version, enricher, fingerprint), indices in sorted(scenes.items())
        ]

    def list_for_scenes(
        self,
        scene_set_version: str,
        scene_indices: Iterable[int],
        pins: Sequence[models.EnrichmentPin] = (),
    ) -> Sequence[models.EnrichmentView]:
        wanted = set(scene_indices)
        self.queries.append((scene_set_version, frozenset(wanted), tuple(pins)))
        return sorted(
            (
                record
                for record in self.records.values()
                if record.scene_set_version == scene_set_version
                and record.scene_index in wanted
                and (not pins or record.pin in pins)
            ),
            key=lambda r: (r.scene_index, r.enricher, r.fingerprint),
        )


class FakeDerivativeReader:
    """DerivativeReader over a dictionary of bytes keyed by digest."""

    def __init__(self, blobs: Mapping[str, bytes] | None = None) -> None:
        self.blobs: dict[str, bytes] = dict(blobs or {})
        self.reads: list[str] = []

    def read(self, reference: models.DerivativeRefView) -> bytes:
        self.reads.append(reference.digest)
        try:
            return self.blobs[reference.digest]
        except KeyError:
            raise DerivativeUnavailableError(f"{reference.uri}: missing") from None


def blobs_of(*scene_sets: models.SceneSetView) -> dict[str, bytes]:
    """The bytes behind every derivative the builders reference."""
    blobs: dict[str, bytes] = {}
    for scene_set in scene_sets:
        for scene in scene_set.scenes:
            for reference in scene.derivatives:
                is_image = reference.media_type == "image/jpeg"
                blobs[reference.digest] = b"jpeg" if is_image else b"wav"
    return blobs


class FakeSceneCatalog:
    """SceneCatalog answering from scripted state, recording every call.

    Set `failure` to an exception to raise it from every call instead.
    """

    def __init__(
        self,
        *,
        scene_sets: Sequence[models.SceneSetSummary] = (),
        records: Sequence[models.SceneRecord] = (),
        assets: Sequence[models.AssetVersionView] = (),
        derivatives: Mapping[str, models.DerivativeContent] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.scene_sets = {s.version: s for s in scene_sets}
        self.records = {(r.scene_set_version, r.scene.index): r for r in records}
        self.assets = {a.version: a for a in assets}
        self.derivatives = dict(derivatives or {})
        self.failure = failure
        self.calls: list[tuple[str, object]] = []

    def _record_call(self, name: str, argument: object) -> None:
        self.calls.append((name, argument))
        if self.failure is not None:
            raise self.failure

    def list_scene_sets(self, query: models.SceneSetQuery) -> models.Page[models.SceneSetSummary]:
        self._record_call("list_scene_sets", query)
        if query.limit is not None and query.limit < 1:
            raise InvalidQueryError("limit must be at least 1")
        return models.Page(tuple(self.scene_sets.values()), "cursor-1")

    def get_scene_set(self, version: str) -> models.SceneSetSummary:
        self._record_call("get_scene_set", version)
        if version not in self.scene_sets:
            raise NotFoundError("scene-set", version)
        return self.scene_sets[version]

    def list_scene_records(
        self, version: str, query: models.SceneQuery
    ) -> models.Page[models.SceneRecord]:
        self._record_call("list_scene_records", (version, query))
        items = tuple(r for (v, _), r in sorted(self.records.items()) if v == version)
        if not items:
            raise NotFoundError("scene-set", version)
        return models.Page(items, None)

    def get_scene_record(
        self, version: str, index: int, pins: Sequence[models.EnrichmentPin]
    ) -> models.SceneRecord:
        self._record_call("get_scene_record", (version, index, tuple(pins)))
        if (version, index) not in self.records:
            raise NotFoundError("scene", f"{version}/{index}")
        return self.records[(version, index)]

    def get_asset_version(self, version: str) -> models.AssetVersionView:
        self._record_call("get_asset_version", version)
        if version not in self.assets:
            raise NotFoundError("asset-version", version)
        return self.assets[version]

    def find_derivative(self, digest: str) -> models.DerivativeRefView:
        self._record_call("find_derivative", digest)
        if digest not in self.derivatives:
            raise NotFoundError("derivative", digest)
        return self.derivatives[digest].reference

    def get_derivative(self, digest: str) -> models.DerivativeContent:
        self._record_call("get_derivative", digest)
        if digest not in self.derivatives:
            raise NotFoundError("derivative", digest)
        return self.derivatives[digest]


@dataclasses.dataclass
class SceneApiHarness:
    """A scene catalog service wired to fakes, exposing them for assertions."""

    assets: FakeAssetVersionReader
    scene_sets: FakeSceneSetReader
    enrichments: FakeEnrichmentReader
    derivatives: FakeDerivativeReader
    page_size: int
    page_fill: int
    service: SceneCatalogService

    def seed(
        self,
        *scene_sets: models.SceneSetView,
        live: bool = True,
        enrichments: Iterable[models.EnrichmentView] = (),
    ) -> None:
        """Register the scene sets, their asset versions, derivative bytes and enrichments."""
        for scene_set in scene_sets:
            self.scene_sets.scene_sets[scene_set.version] = scene_set
            self.assets.records.setdefault(
                scene_set.asset_version, make_asset_view(scene_set.asset_version)
            )
            if not live:
                self.assets.retire(scene_set.asset_version, T0 + timedelta(hours=1))
        self.derivatives.blobs.update(blobs_of(*scene_sets))
        self.enrichments.add(*enrichments)


def make_scene_api_harness(page_size: int = 100, page_fill: int = 1) -> SceneApiHarness:
    assets = FakeAssetVersionReader()
    scene_sets = FakeSceneSetReader()
    enrichments = FakeEnrichmentReader()
    derivatives = FakeDerivativeReader()
    service = SceneCatalogService(
        assets=assets,
        scene_sets=scene_sets,
        enrichments=enrichments,
        derivatives=derivatives,
        page_size=page_size,
        page_fill=page_fill,
    )
    return SceneApiHarness(
        assets, scene_sets, enrichments, derivatives, page_size, page_fill, service
    )
