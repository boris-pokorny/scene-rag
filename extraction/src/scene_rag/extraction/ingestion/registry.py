"""Asset Registry backend over MongoDB.

`MongoAssetRegistry` implements `AssetRegistryReader` and `AssetRegistryWriter`. Every
command is one database operation, so concurrent workers cannot lose each other's writes.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from scene_rag.extraction.ingestion import models
from scene_rag.extraction.ingestion.protocols import UnknownVersionError

COLLECTION: Final = "asset_versions"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def reference_to_document(reference: models.SourceReference) -> Document:
    """Encode a source reference as stored inside a registry document."""
    return {
        "source_id": reference.source_id,
        "asset_id": reference.asset_id,
        "uri": reference.uri,
        "filename": reference.filename,
        "size_bytes": reference.size_bytes,
        "modified_at": reference.modified_at,
        "entitlements": sorted(reference.entitlements),
        "seen_at": reference.seen_at,
    }


def reference_from_document(document: Document) -> models.SourceReference:
    """Decode a source reference stored inside a registry document."""
    return models.SourceReference(
        source_id=document["source_id"],
        asset_id=document["asset_id"],
        uri=document["uri"],
        filename=document["filename"],
        size_bytes=document["size_bytes"],
        modified_at=_utc(document["modified_at"]),
        entitlements=frozenset(document["entitlements"]),
        seen_at=_utc(document["seen_at"]),
    )


def to_document(record: models.AssetVersion) -> Document:
    """Encode an asset version as a registry document (``contracts/internal``)."""
    return {
        "_id": record.version,
        "schema_version": record.schema_version,
        "media_type": record.media_type,
        "size_bytes": record.size_bytes,
        "sources": [reference_to_document(reference) for reference in record.sources],
        "first_seen_at": record.first_seen_at,
        "lineage": {
            "step": record.lineage.step,
            "fingerprint": record.lineage.fingerprint,
            "inputs": list(record.lineage.inputs),
            "version": record.lineage.version,
            "image": record.lineage.image,
        },
        "retired_at": record.retired_at,
    }


def from_document(document: Document) -> models.AssetVersion:
    """Decode a registry document into an asset version."""
    lineage = document["lineage"]
    retired_at = document["retired_at"]
    return models.AssetVersion(
        version=document["_id"],
        media_type=document["media_type"],
        size_bytes=document["size_bytes"],
        sources=tuple(reference_from_document(source) for source in document["sources"]),
        first_seen_at=_utc(document["first_seen_at"]),
        lineage=models.Lineage(
            step=lineage["step"],
            fingerprint=lineage["fingerprint"],
            inputs=tuple(lineage["inputs"]),
            version=lineage["version"],
            image=lineage["image"],
        ),
        retired_at=None if retired_at is None else _utc(retired_at),
        schema_version=document["schema_version"],
    )


def _without_reference(source_id: str, asset_id: str) -> Document:
    """Aggregation expression: ``$sources`` minus the reference with the given key."""
    return {
        "$filter": {
            "input": "$sources",
            "cond": {
                "$not": [
                    {
                        "$and": [
                            {"$eq": ["$$this.source_id", source_id]},
                            {"$eq": ["$$this.asset_id", asset_id]},
                        ]
                    }
                ]
            },
        }
    }


class MongoAssetRegistry:
    """Asset Registry over one MongoDB collection, ``_id`` being the version."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to ``database`` on ``client`` and ensure the indexes exist."""
        self._collection = client[database][COLLECTION]
        self._collection.create_index(
            [("sources.source_id", ASCENDING), ("sources.asset_id", ASCENDING)]
        )
        self._collection.create_index([("retired_at", ASCENDING)])

    def get(self, version: str) -> models.AssetVersion | None:
        """Return the version with the given content hash, live or retired."""
        document = self._collection.find_one({"_id": version})
        return None if document is None else from_document(document)

    def list_by_source(self, source_id: str) -> Sequence[models.AssetVersion]:
        """Return every live version holding at least one reference from the source."""
        cursor = self._collection.find({"retired_at": None, "sources.source_id": source_id}).sort(
            "_id", ASCENDING
        )
        return [from_document(document) for document in cursor]

    def register_version(self, record: models.AssetVersion) -> bool:
        """Insert the record; ``False`` when its version already exists."""
        try:
            self._collection.insert_one(to_document(record))
        except DuplicateKeyError:
            return False
        return True

    def add_source_reference(self, version: str, reference: models.SourceReference) -> None:
        """Replace any reference with the same key, append this one, clear the tombstone."""
        result = self._collection.update_one(
            {"_id": version},
            [
                {
                    "$set": {
                        "sources": {
                            "$concatArrays": [
                                _without_reference(reference.source_id, reference.asset_id),
                                [reference_to_document(reference)],
                            ]
                        },
                        "retired_at": None,
                    }
                }
            ],
        )
        if result.matched_count == 0:
            raise UnknownVersionError(version)

    def remove_source_reference(
        self, version: str, source_id: str, asset_id: str, *, at: datetime
    ) -> models.AssetVersion:
        """Pull the reference; tombstone the version when no reference remains."""
        document = self._collection.find_one_and_update(
            {"_id": version},
            [
                {"$set": {"sources": _without_reference(source_id, asset_id)}},
                {
                    "$set": {
                        "retired_at": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$eq": [{"$size": "$sources"}, 0]},
                                        {"$eq": ["$retired_at", None]},
                                    ]
                                },
                                at,
                                "$retired_at",
                            ]
                        }
                    }
                },
            ],
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise UnknownVersionError(version)
        return from_document(document)
