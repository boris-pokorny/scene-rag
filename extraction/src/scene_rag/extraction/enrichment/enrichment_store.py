"""Enrichment Store backend over MongoDB.

`MongoEnrichmentStore` implements `EnrichmentStoreReader` and `EnrichmentStoreWriter`. A record
is inserted once under its data version and never updated.
"""

from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from scene_rag.extraction.enrichment import models

COLLECTION: Final = "enrichments"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def segment_to_document(segment: models.SemanticSegment) -> Document:
    """Encode a semantic segment as stored inside a record."""
    return {
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "text": segment.text,
        "language": segment.language,
        "confidence": segment.confidence,
    }


def segment_from_document(document: Document) -> models.SemanticSegment:
    """Decode a semantic segment stored inside a record."""
    return models.SemanticSegment(
        start_seconds=document["start_seconds"],
        end_seconds=document["end_seconds"],
        text=document["text"],
        language=document["language"],
        confidence=document["confidence"],
    )


def to_document(record: models.Enrichment) -> Document:
    """Encode a record as a store document (``contracts/internal``)."""
    return {
        "_id": record.version,
        "schema_version": record.schema_version,
        "scene_set_version": record.scene_set_version,
        "asset_version": record.asset_version,
        "scene_index": record.scene_index,
        "enricher": record.enricher,
        "classification": record.classification,
        "model": {"name": record.model.name, "digest": record.model.digest},
        "segments": [segment_to_document(segment) for segment in record.segments],
        "created_at": record.created_at,
        "lineage": {
            "step": record.lineage.step,
            "fingerprint": record.lineage.fingerprint,
            "inputs": list(record.lineage.inputs),
            "version": record.lineage.version,
            "image": record.lineage.image,
        },
    }


def from_document(document: Document) -> models.Enrichment:
    """Decode a store document into a record."""
    lineage = document["lineage"]
    return models.Enrichment(
        version=document["_id"],
        scene_set_version=document["scene_set_version"],
        asset_version=document["asset_version"],
        scene_index=document["scene_index"],
        enricher=document["enricher"],
        classification=document["classification"],
        model=models.ModelRef(name=document["model"]["name"], digest=document["model"]["digest"]),
        segments=tuple(segment_from_document(segment) for segment in document["segments"]),
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


class MongoEnrichmentStore:
    """Enrichment Store over one MongoDB collection, ``_id`` being the record's data version."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to ``database`` on ``client`` and ensure the indexes exist.

        The compound ``(enricher, lineage.fingerprint, scene_set_version, scene_index)`` index
        serves `list_done` directly; ``asset_version`` and ``scene_set_version`` serve the
        liveness, purge and catalog queries the contract promises consumers.
        """
        self._collection = client[database][COLLECTION]
        self._collection.create_index(
            [
                ("enricher", ASCENDING),
                ("lineage.fingerprint", ASCENDING),
                ("scene_set_version", ASCENDING),
                ("scene_index", ASCENDING),
            ]
        )
        self._collection.create_index([("asset_version", ASCENDING)])
        self._collection.create_index([("scene_set_version", ASCENDING)])

    def get(self, version: str) -> models.Enrichment | None:
        """Return the record with the given data version."""
        document = self._collection.find_one({"_id": version})
        return None if document is None else from_document(document)

    def list_done(self, enricher: str, fingerprint: str) -> frozenset[tuple[str, int]]:
        """Return the scenes the enricher has a record for under the fingerprint."""
        cursor = self._collection.find(
            {"enricher": enricher, "lineage.fingerprint": fingerprint},
            projection={"_id": False, "scene_set_version": True, "scene_index": True},
        )
        return frozenset(
            (document["scene_set_version"], document["scene_index"]) for document in cursor
        )

    def put(self, record: models.Enrichment) -> bool:
        """Insert the record; ``False`` when its version already exists."""
        try:
            self._collection.insert_one(to_document(record))
        except DuplicateKeyError:
            return False
        return True
