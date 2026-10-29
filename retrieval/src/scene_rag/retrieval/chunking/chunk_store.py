"""Chunk Store backend over MongoDB.

`MongoChunkStore` implements `ChunkStoreReader` and `ChunkStoreWriter` over two collections:
``chunks`` holds the retrieval units, ``chunk_sets`` the completion marker of each scene set
under each fingerprint. A record is inserted once under its data version and never updated.
"""

from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from scene_rag.retrieval.chunking import models

CHUNKS_COLLECTION: Final = "chunks"
CHUNK_SETS_COLLECTION: Final = "chunk_sets"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def lineage_to_document(lineage: models.Lineage) -> Document:
    """Encode a lineage as stored inside a record."""
    return {
        "step": lineage.step,
        "fingerprint": lineage.fingerprint,
        "inputs": list(lineage.inputs),
        "version": lineage.version,
        "image": lineage.image,
    }


def lineage_from_document(document: Document) -> models.Lineage:
    """Decode a lineage stored inside a record."""
    return models.Lineage(
        step=document["step"],
        fingerprint=document["fingerprint"],
        inputs=tuple(document["inputs"]),
        version=document["version"],
        image=document["image"],
    )


def chunk_to_document(chunk: models.Chunk) -> Document:
    """Encode a chunk as a store document (``contracts/internal/chunk.schema.json``)."""
    return {
        "_id": chunk.version,
        "schema_version": chunk.schema_version,
        "scene_set_version": chunk.scene_set_version,
        "asset_version": chunk.asset_version,
        "segmentation_fingerprint": chunk.segmentation_fingerprint,
        "chunk_index": chunk.chunk_index,
        "scene_indices": list(chunk.scene_indices),
        "start_seconds": chunk.start_seconds,
        "end_seconds": chunk.end_seconds,
        "text": chunk.text,
        "sources": [
            {"source_id": s.source_id, "asset_id": s.asset_id, "filename": s.filename}
            for s in chunk.sources
        ],
        "entitlements": sorted(chunk.entitlements),
        "classifications": list(chunk.classifications),
        "keyframes": [{"digest": k.digest, "seconds": k.seconds} for k in chunk.keyframes],
        "enrichments": [
            {"enricher": e.enricher, "fingerprint": e.fingerprint, "version": e.version}
            for e in chunk.enrichments
        ],
        "created_at": chunk.created_at,
        "lineage": lineage_to_document(chunk.lineage),
    }


def chunk_from_document(document: Document) -> models.Chunk:
    """Decode a store document into a chunk."""
    return models.Chunk(
        version=document["_id"],
        scene_set_version=document["scene_set_version"],
        asset_version=document["asset_version"],
        segmentation_fingerprint=document["segmentation_fingerprint"],
        chunk_index=document["chunk_index"],
        scene_indices=tuple(document["scene_indices"]),
        start_seconds=document["start_seconds"],
        end_seconds=document["end_seconds"],
        text=document["text"],
        sources=tuple(
            models.SourceView(s["source_id"], s["asset_id"], s["filename"])
            for s in document["sources"]
        ),
        entitlements=frozenset(document["entitlements"]),
        classifications=tuple(document["classifications"]),
        keyframes=tuple(
            models.KeyframeRef(k["digest"], k["seconds"]) for k in document["keyframes"]
        ),
        enrichments=tuple(
            models.EnrichmentRef(e["enricher"], e["fingerprint"], e["version"])
            for e in document["enrichments"]
        ),
        created_at=_utc(document["created_at"]),
        lineage=lineage_from_document(document["lineage"]),
        schema_version=document["schema_version"],
    )


def chunk_set_to_document(chunk_set: models.ChunkSet) -> Document:
    """Encode a chunk set as a store document (``contracts/internal/chunk-set.schema.json``)."""
    return {
        "_id": chunk_set.version,
        "schema_version": chunk_set.schema_version,
        "scene_set_version": chunk_set.scene_set_version,
        "asset_version": chunk_set.asset_version,
        "segmentation_fingerprint": chunk_set.segmentation_fingerprint,
        "pins": [str(pin) for pin in chunk_set.pins],
        "chunk_count": chunk_set.chunk_count,
        "created_at": chunk_set.created_at,
        "lineage": lineage_to_document(chunk_set.lineage),
    }


def chunk_set_from_document(document: Document) -> models.ChunkSet:
    """Decode a store document into a chunk set."""
    return models.ChunkSet(
        version=document["_id"],
        scene_set_version=document["scene_set_version"],
        asset_version=document["asset_version"],
        segmentation_fingerprint=document["segmentation_fingerprint"],
        pins=tuple(models.EnrichmentPin.parse(pin) for pin in document["pins"]),
        chunk_count=document["chunk_count"],
        created_at=_utc(document["created_at"]),
        lineage=lineage_from_document(document["lineage"]),
        schema_version=document["schema_version"],
    )


class MongoChunkStore:
    """Chunk Store over two MongoDB collections, ``_id`` being each record's data version."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to ``database`` on ``client`` and ensure the indexes exist.

        ``(lineage.fingerprint, scene_set_version)`` serves `list_done` and the per-set reads
        of the next step; ``asset_version`` serves the purge of retired assets.
        """
        self._chunks = client[database][CHUNKS_COLLECTION]
        self._chunk_sets = client[database][CHUNK_SETS_COLLECTION]
        for collection in (self._chunks, self._chunk_sets):
            collection.create_index(
                [("lineage.fingerprint", ASCENDING), ("scene_set_version", ASCENDING)]
            )
            collection.create_index([("asset_version", ASCENDING)])

    def get_chunk(self, version: str) -> models.Chunk | None:
        """Return the chunk with the given data version."""
        document = self._chunks.find_one({"_id": version})
        return None if document is None else chunk_from_document(document)

    def get_chunk_set(self, version: str) -> models.ChunkSet | None:
        """Return the chunk set with the given data version."""
        document = self._chunk_sets.find_one({"_id": version})
        return None if document is None else chunk_set_from_document(document)

    def list_done(self, fingerprint: str) -> frozenset[str]:
        """Return the scene set versions with a chunk set under the fingerprint."""
        cursor = self._chunk_sets.find(
            {"lineage.fingerprint": fingerprint},
            projection={"_id": False, "scene_set_version": True},
        )
        return frozenset(document["scene_set_version"] for document in cursor)

    def put_chunk(self, chunk: models.Chunk) -> bool:
        """Insert the chunk; ``False`` when its version already exists."""
        try:
            self._chunks.insert_one(chunk_to_document(chunk))
        except DuplicateKeyError:
            return False
        return True

    def put_chunk_set(self, chunk_set: models.ChunkSet) -> bool:
        """Insert the chunk set; ``False`` when its version already exists."""
        try:
            self._chunk_sets.insert_one(chunk_set_to_document(chunk_set))
        except DuplicateKeyError:
            return False
        return True
