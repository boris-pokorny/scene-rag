"""Read model over the Asset Registry.

`MongoAssetVersionReader` reads the ``asset_versions`` collection ingestion writes, decoding
only the fields enrichment needs: whether a version is still live. The document shape is
pinned by ``contracts/internal/asset-version.schema.json``; the step keeps its own copy of the
model rather than importing a sibling's, so the steps share a contract, not code.
"""

from datetime import UTC, datetime
from typing import Any, Final

from pymongo import MongoClient

from scene_rag.extraction.enrichment import models

COLLECTION: Final = "asset_versions"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]

_LIVE_FILTER: Final = {"retired_at": None, "sources.0": {"$exists": True}}


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def view_from_document(document: Document) -> models.AssetVersionView:
    """Decode a registry document into the step's view of the version."""
    retired_at = document["retired_at"]
    return models.AssetVersionView(
        version=document["_id"],
        has_sources=bool(document["sources"]),
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

    def list_live_versions(self) -> frozenset[str]:
        """Return the content hashes of every version that is not retired and has a source."""
        cursor = self._collection.find(_LIVE_FILTER, projection={"_id": True})
        return frozenset(document["_id"] for document in cursor)
