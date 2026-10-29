"""Read model over the Chunk Store: the chunking step's collections, as embedding sees them.

`MongoChunkReader` implements `ChunkReader` over the ``chunks`` and ``chunk_sets``
collections the chunking step writes. It decodes only what embedding needs and is pinned by
``contracts/internal/chunk.schema.json`` and ``chunk-set.schema.json``, not by shared code:
steps never import each other. It never writes.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient

from scene_rag.retrieval.embedding import models

CHUNKS_COLLECTION: Final = "chunks"
CHUNK_SETS_COLLECTION: Final = "chunk_sets"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _check_schema(document: Document, name: str) -> None:
    version = document.get("schema_version")
    if version != models.CHUNK_SCHEMA_VERSION:
        raise ValueError(
            f"{name} {document.get('_id')!r} has schema_version {version!r}; "
            f"this step reads version {models.CHUNK_SCHEMA_VERSION}"
        )


def chunk_from_document(document: Document) -> models.ChunkView:
    """Decode a ``chunks`` document (``contracts/internal/chunk.schema.json``).

    Raises:
        ValueError: When the document's schema version is not the one this step reads.
    """
    _check_schema(document, "chunk")
    return models.ChunkView(
        version=document["_id"],
        scene_set_version=document["scene_set_version"],
        asset_version=document["asset_version"],
        segmentation_fingerprint=document["segmentation_fingerprint"],
        chunking_fingerprint=document["lineage"]["fingerprint"],
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
    )


def chunk_set_from_document(document: Document) -> models.ChunkSetView:
    """Decode a ``chunk_sets`` document (``contracts/internal/chunk-set.schema.json``).

    Raises:
        ValueError: When the document's schema version is not the one this step reads.
    """
    _check_schema(document, "chunk set")
    return models.ChunkSetView(
        version=document["_id"],
        scene_set_version=document["scene_set_version"],
        asset_version=document["asset_version"],
        chunking_fingerprint=document["lineage"]["fingerprint"],
        chunk_count=document["chunk_count"],
        created_at=_utc(document["created_at"]),
    )


class MongoChunkReader:
    """`ChunkReader` over the chunking step's collections; the index it needs exists already.

    ``(lineage.fingerprint, scene_set_version)`` is created by the writer of the collections;
    this reader creates nothing.
    """

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to ``database`` on ``client``."""
        self._chunks = client[database][CHUNKS_COLLECTION]
        self._chunk_sets = client[database][CHUNK_SETS_COLLECTION]

    def list_chunk_sets(self, chunking_fingerprint: str | None) -> Iterator[models.ChunkSetView]:
        """Yield the chunk sets under the fingerprint (every one when ``None``), by version."""
        query: Document = {}
        if chunking_fingerprint is not None:
            query["lineage.fingerprint"] = chunking_fingerprint
        for document in self._chunk_sets.find(query).sort("_id", ASCENDING):
            yield chunk_set_from_document(document)

    def list_chunks(
        self, scene_set_version: str, chunking_fingerprint: str
    ) -> list[models.ChunkView]:
        """Return the chunks of the scene set under the fingerprint, in index order."""
        cursor = self._chunks.find(
            {"lineage.fingerprint": chunking_fingerprint, "scene_set_version": scene_set_version}
        ).sort("chunk_index", ASCENDING)
        return [chunk_from_document(document) for document in cursor]
