"""Read model over the Scene Store.

`MongoSceneSetReader` reads the ``scene_sets`` collection segmentation writes. The document
shape is pinned by ``contracts/internal/scene-set.schema.json``; the role keeps its own view
rather than importing segmentation's model, so the two share a contract, not code. The reader
creates no index: the owner does, including the ones these queries rely on.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient

from scene_rag.extraction.scene_api import models

COLLECTION: Final = "scene_sets"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]

_SUMMARY_PROJECTION: Final = {
    "asset_version": True,
    "lineage.fingerprint": True,
    "created_at": True,
    "scene_count": {"$size": "$scenes"},
}


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def derivative_from_document(document: Document) -> models.DerivativeRefView:
    """Decode a derivative reference stored inside a scene set document."""
    return models.DerivativeRefView(
        digest=document["digest"],
        uri=document["uri"],
        media_type=document["media_type"],
        size_bytes=document["size_bytes"],
    )


def scene_from_document(document: Document) -> models.SceneView:
    """Decode a scene stored inside a scene set document."""
    audio_clip = document["audio_clip"]
    return models.SceneView(
        index=document["index"],
        start_seconds=document["start_seconds"],
        end_seconds=document["end_seconds"],
        start_frame=document["start_frame"],
        end_frame=document["end_frame"],
        keyframes=tuple(
            models.KeyframeView(
                derivative=derivative_from_document(keyframe["derivative"]),
                frame=keyframe["frame"],
                seconds=keyframe["seconds"],
            )
            for keyframe in document["keyframes"]
        ),
        audio_clip=None if audio_clip is None else derivative_from_document(audio_clip),
    )


def media_from_document(document: Document) -> models.MediaView:
    """Decode the media properties stored inside a scene set document."""
    return models.MediaView(
        duration_seconds=document["duration_seconds"],
        frame_rate=document["frame_rate"],
        width=document["width"],
        height=document["height"],
        frame_count=document["frame_count"],
    )


def scene_set_from_document(document: Document) -> models.SceneSetView:
    """Decode a scene set document into the role's view of it."""
    return models.SceneSetView(
        version=document["_id"],
        asset_version=document["asset_version"],
        segmentation_fingerprint=document["lineage"]["fingerprint"],
        media=media_from_document(document["media"]),
        scenes=tuple(scene_from_document(scene) for scene in document["scenes"]),
        created_at=_utc(document["created_at"]),
    )


def summary_from_document(document: Document) -> models.SceneSetSummaryView:
    """Decode a projected scene set document (``scene_count`` in place of ``scenes``)."""
    return models.SceneSetSummaryView(
        version=document["_id"],
        asset_version=document["asset_version"],
        segmentation_fingerprint=document["lineage"]["fingerprint"],
        scene_count=document["scene_count"],
        created_at=_utc(document["created_at"]),
    )


class MongoSceneSetReader:
    """`SceneSetReader` over the scene set collection; never writes to it."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to the scene set collection in ``database`` on ``client``."""
        self._collection = client[database][COLLECTION]

    def get(self, version: str) -> models.SceneSetView | None:
        """Return the scene set with the given data version."""
        document = self._collection.find_one({"_id": version})
        return None if document is None else scene_set_from_document(document)

    def list_summaries(
        self,
        *,
        after: str | None,
        limit: int,
        asset_version: str | None,
        segmentation_fingerprint: str | None,
    ) -> Sequence[models.SceneSetSummaryView]:
        """Return up to ``limit`` summaries in version order, those after the cursor."""
        criteria: Document = {}
        if after is not None:
            criteria["_id"] = {"$gt": after}
        if asset_version is not None:
            criteria["asset_version"] = asset_version
        if segmentation_fingerprint is not None:
            criteria["lineage.fingerprint"] = segmentation_fingerprint
        cursor = (
            self._collection.find(criteria, projection=_SUMMARY_PROJECTION)
            .sort("_id", ASCENDING)
            .limit(limit)
        )
        return [summary_from_document(document) for document in cursor]

    def find_derivative(self, digest: str) -> models.DerivativeRefView | None:
        """Return the reference some scene set holds to the derivative with this digest."""
        document = self._collection.find_one(
            {
                "$or": [
                    {"scenes.keyframes.derivative.digest": digest},
                    {"scenes.audio_clip.digest": digest},
                ]
            },
            projection={"scenes": True},
        )
        if document is None:
            return None
        for scene in document["scenes"]:
            for reference in scene_from_document(scene).derivatives:
                if reference.digest == digest:
                    return reference
        return None  # pragma: no cover - the query matched, so a reference is there
