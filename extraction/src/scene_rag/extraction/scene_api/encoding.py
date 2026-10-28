"""JSON encoding of what the role publishes: the shapes of ``contracts/published``.

The only module naming the published field names; the contract tests pin it to the schemas.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from scene_rag.extraction.scene_api import models

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]

DERIVATIVES_PATH = "/derivatives"


def encode_datetime(moment: datetime) -> str:
    """Render an aware timestamp as RFC 3339 in UTC with a ``Z`` suffix.

    A naive value is taken as local time, as `datetime.astimezone` does; the readers attach UTC
    to what the stores return before it gets here.
    """
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def derivative_url(digest: str) -> str:
    """Return the path the bytes of a derivative are served at."""
    return f"{DERIVATIVES_PATH}/{digest}"


def encode_derivative(reference: models.DerivativeRefView) -> Json:
    """Encode a derivative reference, with the API path in place of the internal URI."""
    return {
        "digest": reference.digest,
        "media_type": reference.media_type,
        "size_bytes": reference.size_bytes,
        "url": derivative_url(reference.digest),
    }


def encode_keyframe(keyframe: models.KeyframeView) -> Json:
    """Encode a keyframe: its derivative, flattened, with its position."""
    return {
        **encode_derivative(keyframe.derivative),
        "frame": keyframe.frame,
        "seconds": keyframe.seconds,
    }


def encode_scene(scene: models.SceneView) -> Json:
    """Encode a scene with its derivative references."""
    return {
        "index": scene.index,
        "start_seconds": scene.start_seconds,
        "end_seconds": scene.end_seconds,
        "start_frame": scene.start_frame,
        "end_frame": scene.end_frame,
        "keyframes": [encode_keyframe(keyframe) for keyframe in scene.keyframes],
        "audio_clip": None if scene.audio_clip is None else encode_derivative(scene.audio_clip),
    }


def encode_media(media: models.MediaView) -> Json:
    """Encode the properties of the video stream."""
    return {
        "duration_seconds": media.duration_seconds,
        "frame_rate": media.frame_rate,
        "width": media.width,
        "height": media.height,
        "frame_count": media.frame_count,
    }


def encode_source(source: models.SourceView) -> Json:
    """Encode a source reference: ids and display name, never the connector's location."""
    return {
        "source_id": source.source_id,
        "asset_id": source.asset_id,
        "filename": source.filename,
    }


def encode_asset(asset: models.AssetVersionView) -> Json:
    """Encode what a record says of its asset version: type, size, entitlements, source ids."""
    return {
        "media_type": asset.media_type,
        "size_bytes": asset.size_bytes,
        "entitlements": sorted(asset.entitlements),
        "sources": [encode_source(source) for source in asset.sources],
    }


def encode_asset_version(asset: models.AssetVersionView) -> Json:
    """Encode an asset version as ``asset-version.schema.json`` describes it."""
    return {
        "version": asset.version,
        "live": asset.is_live,
        "retired_at": None if asset.retired_at is None else encode_datetime(asset.retired_at),
        **encode_asset(asset),
    }


def encode_segment(segment: models.SemanticSegmentView) -> Json:
    """Encode a semantic segment."""
    return {
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "text": segment.text,
        "language": segment.language,
        "confidence": segment.confidence,
    }


def encode_enrichment(record: models.EnrichmentView) -> Json:
    """Encode one enrichment of a scene record."""
    return {
        "version": record.version,
        "enricher": record.enricher,
        "fingerprint": record.fingerprint,
        "classification": record.classification,
        "model": {"name": record.model.name, "digest": record.model.digest},
        "segments": [encode_segment(segment) for segment in record.segments],
    }


def encode_scene_record(record: models.SceneRecord) -> Json:
    """Encode a scene record as ``scene-record.schema.json`` describes it."""
    return {
        "schema_version": record.schema_version,
        "version": record.version,
        "scene_set_version": record.scene_set_version,
        "asset_version": record.asset_version,
        "segmentation_fingerprint": record.segmentation_fingerprint,
        "live": record.live,
        "asset": encode_asset(record.asset),
        "media": encode_media(record.media),
        "scene": encode_scene(record.scene),
        "enrichments": [encode_enrichment(item) for item in record.enrichments],
    }


def encode_scene_set_summary(summary: models.SceneSetSummary) -> Json:
    """Encode a scene set summary as ``scene-set-summary.schema.json`` describes it."""
    return {
        "version": summary.version,
        "asset_version": summary.asset_version,
        "segmentation_fingerprint": summary.segmentation_fingerprint,
        "scene_count": summary.scene_count,
        "created_at": encode_datetime(summary.created_at),
        "live": summary.live,
        "enrichments": [
            {
                "enricher": item.enricher,
                "fingerprint": item.fingerprint,
                "scene_count": item.scene_count,
                "complete": item.complete,
            }
            for item in summary.enrichments
        ],
    }


def encode_page[T](page: models.Page[T], encode_item: Callable[[T], Json]) -> Json:
    """Encode a page as ``page.schema.json`` describes it."""
    return {"items": [encode_item(item) for item in page.items], "next": page.next}


def encode_error(code: str, message: str) -> Json:
    """Encode an error body as ``error.schema.json`` describes it."""
    return {"error": {"code": code, "message": message}}
