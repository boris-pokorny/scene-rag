"""Read model over the Asset Registry.

`MongoAssetVersionReader` reads the ``asset_versions`` collection ingestion writes, decoding
only the fields segmentation needs. The document shape is pinned by
``contracts/internal/asset-version.schema.json``; the step keeps its own copy of the model
rather than importing ingestion's, so the two steps share a contract, not code.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient

from scene_rag.extraction.segmentation import models

COLLECTION: Final = "asset_versions"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def view_from_document(document: Document) -> models.AssetVersionView:
    """Decode a registry document into the step's view of the version."""
    retired_at = document["retired_at"]
    return models.AssetVersionView(
        version=document["_id"],
        media_type=document["media_type"],
        size_bytes=document["size_bytes"],
        sources=tuple(
            models.SourceView(
                source_id=source["source_id"],
                asset_id=source["asset_id"],
                uri=source["uri"],
                size_bytes=source["size_bytes"],
                modified_at=_utc(source["modified_at"]),
            )
            for source in document["sources"]
        ),
        retired_at=None if retired_at is None else _utc(retired_at),
    )


class MongoAssetVersionReader:
    """`AssetVersionReader` over the registry collection; never writes to it."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to the registry collection in ``database`` on ``client``."""
        self._collection = client[database][COLLECTION]

    def get(self, version: str) -> models.AssetVersionView | None:
        """Return the version with the given content hash, live or retired."""
        document = self._collection.find_one({"_id": version})
        return None if document is None else view_from_document(document)

    def list_live(self) -> Sequence[models.AssetVersionView]:
        """Return every version that is not retired and still has a source, in version order."""
        cursor = self._collection.find({"retired_at": None, "sources.0": {"$exists": True}}).sort(
            "_id", ASCENDING
        )
        return [view_from_document(document) for document in cursor]
