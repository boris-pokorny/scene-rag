"""Read model over the Scene Store.

`MongoSceneSetReader` reads the ``scene_sets`` collection segmentation writes, decoding only
the fields enrichment needs. The document shape is pinned by
``contracts/internal/scene-set.schema.json``; the step keeps its own view rather than importing
segmentation's model, so the two steps share a contract, not code.
"""

from collections.abc import Sequence
from typing import Any, Final

from pymongo import ASCENDING, MongoClient

from scene_rag.extraction.enrichment import models

COLLECTION: Final = "scene_sets"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


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


def scene_set_from_document(document: Document) -> models.SceneSetView:
    """Decode a scene set document into the step's view of it."""
    return models.SceneSetView(
        version=document["_id"],
        asset_version=document["asset_version"],
        scenes=tuple(scene_from_document(scene) for scene in document["scenes"]),
    )


def summary_from_document(document: Document) -> models.SceneSetSummary:
    """Decode a projected scene set document (``scene_count`` in place of ``scenes``)."""
    return models.SceneSetSummary(
        version=document["_id"],
        asset_version=document["asset_version"],
        scene_count=document["scene_count"],
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

    def list_summaries(self) -> Sequence[models.SceneSetSummary]:
        """Return every scene set's version, asset version and scene count, in version order."""
        cursor = self._collection.find(
            {}, projection={"asset_version": True, "scene_count": {"$size": "$scenes"}}
        ).sort("_id", ASCENDING)
        return [summary_from_document(document) for document in cursor]
