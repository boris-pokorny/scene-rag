"""Scene Store backend over MongoDB.

`MongoSceneStore` implements `SceneStoreReader` and `SceneStoreWriter`. A scene set is
inserted once under its data version and never updated.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from scene_rag.extraction.segmentation import models

COLLECTION: Final = "scene_sets"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def derivative_to_document(reference: models.DerivativeRef) -> Document:
    """Encode a derivative reference as stored inside a scene set document."""
    return {
        "digest": reference.digest,
        "uri": reference.uri,
        "media_type": reference.media_type,
        "size_bytes": reference.size_bytes,
    }


def derivative_from_document(document: Document) -> models.DerivativeRef:
    """Decode a derivative reference stored inside a scene set document."""
    return models.DerivativeRef(
        digest=document["digest"],
        uri=document["uri"],
        media_type=document["media_type"],
        size_bytes=document["size_bytes"],
    )


def scene_to_document(scene: models.Scene) -> Document:
    """Encode a scene as stored inside a scene set document."""
    return {
        "index": scene.index,
        "start_seconds": scene.start_seconds,
        "end_seconds": scene.end_seconds,
        "start_frame": scene.start_frame,
        "end_frame": scene.end_frame,
        "keyframes": [
            {
                "derivative": derivative_to_document(keyframe.derivative),
                "frame": keyframe.frame,
                "seconds": keyframe.seconds,
            }
            for keyframe in scene.keyframes
        ],
        "audio_clip": None
        if scene.audio_clip is None
        else derivative_to_document(scene.audio_clip),
    }


def scene_from_document(document: Document) -> models.Scene:
    """Decode a scene stored inside a scene set document."""
    audio_clip = document["audio_clip"]
    return models.Scene(
        index=document["index"],
        start_seconds=document["start_seconds"],
        end_seconds=document["end_seconds"],
        start_frame=document["start_frame"],
        end_frame=document["end_frame"],
        keyframes=tuple(
            models.Keyframe(
                derivative=derivative_from_document(keyframe["derivative"]),
                frame=keyframe["frame"],
                seconds=keyframe["seconds"],
            )
            for keyframe in document["keyframes"]
        ),
        audio_clip=None if audio_clip is None else derivative_from_document(audio_clip),
    )


def to_document(scene_set: models.SceneSet) -> Document:
    """Encode a scene set as a store document (``contracts/internal``)."""
    media = scene_set.media
    return {
        "_id": scene_set.version,
        "schema_version": scene_set.schema_version,
        "asset_version": scene_set.asset_version,
        "media": {
            "duration_seconds": media.duration_seconds,
            "frame_rate": media.frame_rate,
            "width": media.width,
            "height": media.height,
            "frame_count": media.frame_count,
        },
        "scenes": [scene_to_document(scene) for scene in scene_set.scenes],
        "created_at": scene_set.created_at,
        "lineage": {
            "step": scene_set.lineage.step,
            "fingerprint": scene_set.lineage.fingerprint,
            "inputs": list(scene_set.lineage.inputs),
            "version": scene_set.lineage.version,
            "image": scene_set.lineage.image,
        },
    }


def from_document(document: Document) -> models.SceneSet:
    """Decode a store document into a scene set."""
    media = document["media"]
    lineage = document["lineage"]
    return models.SceneSet(
        version=document["_id"],
        asset_version=document["asset_version"],
        media=models.MediaInfo(
            duration_seconds=media["duration_seconds"],
            frame_rate=media["frame_rate"],
            width=media["width"],
            height=media["height"],
            frame_count=media["frame_count"],
        ),
        scenes=tuple(scene_from_document(scene) for scene in document["scenes"]),
        created_at=_utc(document["created_at"]),
        lineage=models.Lineage(
            step=lineage["step"],
            fingerprint=lineage["fingerprint"],
            inputs=tuple(lineage["inputs"]),
            version=lineage["version"],
            image=lineage["image"],
        ),
        schema_version=document["schema_version"],
    )


class MongoSceneStore:
    """Scene Store over one MongoDB collection, ``_id`` being the scene set's data version."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to ``database`` on ``client`` and ensure the indexes exist.

        The compound ``(lineage.fingerprint, asset_version)`` index serves
        `list_asset_versions` directly. The others serve the scene-api role, which reads this
        collection but, as a reader, creates no index: listing an asset version's sets, and
        finding the set that references a derivative by its digest.
        """
        self._collection = client[database][COLLECTION]
        self._collection.create_index(
            [("lineage.fingerprint", ASCENDING), ("asset_version", ASCENDING)]
        )
        self._collection.create_index([("asset_version", ASCENDING)])
        self._collection.create_index([("scenes.keyframes.derivative.digest", ASCENDING)])
        self._collection.create_index([("scenes.audio_clip.digest", ASCENDING)])

    def get(self, version: str) -> models.SceneSet | None:
        """Return the scene set with the given data version."""
        document = self._collection.find_one({"_id": version})
        return None if document is None else from_document(document)

    def list_asset_versions(self, fingerprint: str) -> frozenset[str]:
        """Return the asset versions that have a scene set under the fingerprint."""
        versions: Sequence[str] = self._collection.distinct(
            "asset_version", {"lineage.fingerprint": fingerprint}
        )
        return frozenset(versions)

    def put(self, scene_set: models.SceneSet) -> bool:
        """Insert the scene set; ``False`` when its version already exists."""
        try:
            self._collection.insert_one(to_document(scene_set))
        except DuplicateKeyError:
            return False
        return True
