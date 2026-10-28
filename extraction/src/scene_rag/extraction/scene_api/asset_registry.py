"""Read model over the Asset Registry.

`MongoAssetVersionReader` reads the ``asset_versions`` collection ingestion writes, decoding
the fields the role publishes: type, size, source ids, entitlements and liveness. The document
shape is pinned by ``contracts/internal/asset-version.schema.json``; the role keeps its own
view rather than importing ingestion's model, so the two share a contract, not code.
"""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Final

from pymongo import MongoClient

from scene_rag.extraction.scene_api import models

COLLECTION: Final = "asset_versions"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]

_PROJECTION: Final = {
    "media_type": True,
    "size_bytes": True,
    "retired_at": True,
    "sources.source_id": True,
    "sources.asset_id": True,
    "sources.filename": True,
    "sources.entitlements": True,
}


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def view_from_document(document: Document) -> models.AssetVersionView:
    """Decode a registry document (or its projection) into the role's view of the version."""
    retired_at = document["retired_at"]
    sources = document["sources"]
    return models.AssetVersionView(
        version=document["_id"],
        media_type=document["media_type"],
        size_bytes=document["size_bytes"],
        sources=tuple(
            models.SourceView(
                source_id=source["source_id"],
                asset_id=source["asset_id"],
                filename=source["filename"],
            )
            for source in sources
        ),
        entitlements=frozenset(
            entitlement for source in sources for entitlement in source["entitlements"]
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
        document = self._collection.find_one({"_id": version}, projection=_PROJECTION)
        return None if document is None else view_from_document(document)

    def get_many(self, versions: Iterable[str]) -> Mapping[str, models.AssetVersionView]:
        """Return the versions found among those asked for, keyed by content hash."""
        cursor = self._collection.find({"_id": {"$in": list(versions)}}, projection=_PROJECTION)
        return {document["_id"]: view_from_document(document) for document in cursor}
