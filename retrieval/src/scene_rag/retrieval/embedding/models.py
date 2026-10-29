"""Domain model of the embedding step: read views of the Chunk Store, entries, markers."""

import dataclasses
import hashlib
import uuid
from datetime import datetime
from typing import Final

SCHEMA_VERSION: Final = 1
STEP_NAME: Final = "embedding"
CHUNK_INPUT_PREFIX: Final = "chunk:"
CHUNK_SCHEMA_VERSION: Final = 1
"""The Chunk Store schema this step decodes; see `chunk_reader`."""
COLLECTION_PREFIX: Final = "chunks_"
POINT_NAMESPACE: Final = uuid.UUID("5c0e7a1d-3b2f-4e8a-9d6c-1f4b7a2e9c03")
"""Namespace of the point ids: a backend may demand UUIDs, chunk versions are hashes."""

type Vector = tuple[float, ...]


def truncate_to_millisecond(moment: datetime) -> datetime:
    """Return ``moment`` with its microseconds truncated to the millisecond.

    The stores hold BSON dates, which carry milliseconds. Every timestamp the step produces or
    compares is truncated first, so a round-tripped value equals a fresh one.
    """
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def index_entry_version(chunk_version: str, fingerprint: str, algorithm: str) -> str:
    """Derive the data version of an index entry: ``hash(chunk version + fingerprint)``.

    Args:
        chunk_version: Data version of the chunk the entry was built from.
        fingerprint: Fingerprint of the step.
        algorithm: ``hashlib`` algorithm name.

    Returns:
        ``<algorithm>:<hex digest>``.
    """
    digest = hashlib.new(algorithm, f"{chunk_version}\n{fingerprint}".encode())
    return f"{algorithm}:{digest.hexdigest()}"


def index_set_version(chunk_set_version: str, fingerprint: str, algorithm: str) -> str:
    """Derive the data version of an index set: ``hash(chunk set version + fingerprint)``."""
    digest = hashlib.new(algorithm, f"{chunk_set_version}\n{fingerprint}".encode())
    return f"{algorithm}:{digest.hexdigest()}"


def point_id(chunk_version: str) -> str:
    """Derive the id of a chunk's point: a UUID (version 5) of the chunk version.

    Stable across reruns, so an upsert replaces rather than duplicates; unique within a
    collection, since a collection holds one embedding fingerprint and a chunk version is a
    content hash.
    """
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_version))


def collection_name(fingerprint: str) -> str:
    """Name the collection holding the index of one embedding fingerprint."""
    return COLLECTION_PREFIX + fingerprint


@dataclasses.dataclass(frozen=True)
class SourceView:
    """A source system's reference to an asset version, as a chunk carries it.

    Attributes:
        source_id: Identifier of the connector that found the item.
        asset_id: Stable identifier of the item within its source.
        filename: Display name of the item in its source.
    """

    source_id: str
    asset_id: str
    filename: str


@dataclasses.dataclass(frozen=True)
class ModelRef:
    """A model as the backend resolves it.

    Attributes:
        name: Name the model is known by to its backend.
        digest: Digest of the model's bytes.
    """

    name: str
    digest: str


@dataclasses.dataclass(frozen=True)
class ChunkView:
    """A chunk as the Chunk Store holds it, as far as embedding needs it: the step's input.

    Attributes:
        version: Data version of the chunk.
        scene_set_version: Data version of the scene set the chunk was cut from.
        asset_version: Content hash of the asset.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut the scenes.
        chunking_fingerprint: Fingerprint of the chunking run that wrote the chunk.
        chunk_index: Position among the chunks of the scene set under that fingerprint.
        scene_indices: Scenes the chunk covers, ascending.
        start_seconds: Start of the covered span in the stream.
        end_seconds: End of the covered span in the stream.
        text: The retrieval text.
        sources: Source references holding the asset version, for citations.
        entitlements: Who may see the chunk.
        classifications: Distinct data classifications of the text, sorted.
    """

    version: str
    scene_set_version: str
    asset_version: str
    segmentation_fingerprint: str
    chunking_fingerprint: str
    chunk_index: int
    scene_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    text: str
    sources: tuple[SourceView, ...]
    entitlements: frozenset[str]
    classifications: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ChunkSetView:
    """A chunk set as the Chunk Store holds it: the unit of work.

    Attributes:
        version: Data version of the chunk set.
        scene_set_version: Data version of the scene set.
        asset_version: Content hash of the asset.
        chunking_fingerprint: Fingerprint of the chunking run that wrote it.
        chunk_count: Chunks written under it; zero when no scene had text.
        created_at: When the set was written, in UTC.
    """

    version: str
    scene_set_version: str
    asset_version: str
    chunking_fingerprint: str
    chunk_count: int
    created_at: datetime


@dataclasses.dataclass(frozen=True)
class Lineage:
    """How a record of the step was produced.

    Attributes:
        fingerprint: Fingerprint of the step that wrote the record.
        inputs: Data versions read: the chunks.
        version: Data version of the record, ``hash(inputs + fingerprint)``.
        step: Name of the producing step.
        image: Digest of the service image, when known.
    """

    fingerprint: str
    inputs: tuple[str, ...]
    version: str
    step: str = STEP_NAME
    image: str | None = None


@dataclasses.dataclass(frozen=True)
class IndexEntry:
    """What the Search Index keeps next to a chunk's vector: the point's payload.

    Enough to filter, rank and cite from the index alone; keyframes and enrichment
    references are fetched from the Chunk Store by `chunk_version` when needed.

    Attributes:
        version: Data version of the entry, see `index_entry_version`.
        chunk_version: Data version of the chunk: the join key to the Chunk Store.
        chunk_set_version: Data version of the chunk set the chunk belongs to.
        scene_set_version: Data version of the scene set.
        asset_version: Content hash of the asset.
        chunking_fingerprint: Fingerprint of the chunking run that wrote the chunk.
        segmentation_fingerprint: Fingerprint of the segmentation run that cut the scenes.
        chunk_index: Position among the chunks of the scene set.
        scene_indices: Scenes the chunk covers, ascending.
        start_seconds: Start of the covered span in the stream.
        end_seconds: End of the covered span in the stream.
        text: The retrieval text, as embedded (without the document prefix).
        sources: Source references holding the asset version, for citations.
        entitlements: Who may see the chunk.
        classifications: Distinct data classifications of the text, sorted.
        created_at: When the point was written, in UTC.
        lineage: Provenance of the entry.
        schema_version: Version of the payload's schema.
    """

    version: str
    chunk_version: str
    chunk_set_version: str
    scene_set_version: str
    asset_version: str
    chunking_fingerprint: str
    segmentation_fingerprint: str
    chunk_index: int
    scene_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    text: str
    sources: tuple[SourceView, ...]
    entitlements: frozenset[str]
    classifications: tuple[str, ...]
    created_at: datetime
    lineage: Lineage
    schema_version: int = SCHEMA_VERSION


@dataclasses.dataclass(frozen=True)
class Point:
    """One vector with its entry, as written to the Search Index.

    Attributes:
        id: Identifier of the point in its collection, see `point_id`.
        vector: The embedding.
        entry: The payload.
    """

    id: str
    vector: Vector
    entry: IndexEntry


@dataclasses.dataclass(frozen=True)
class IndexSet:
    """One chunk set embedded under one fingerprint: the completion marker the step writes last.

    Attributes:
        version: Data version and identifier, see `index_set_version`.
        chunk_set_version: Data version of the chunk set.
        scene_set_version: Data version of the scene set.
        asset_version: Content hash of the asset.
        chunking_fingerprint: Fingerprint of the chunking run that wrote the chunk set.
        collection: Collection of the Search Index the points went to.
        vector_count: Points written; zero when the chunk set had no chunks.
        created_at: When the record was written, in UTC.
        lineage: Provenance: every chunk version read, in chunk order.
        schema_version: Version of the stored record's schema.
    """

    version: str
    chunk_set_version: str
    scene_set_version: str
    asset_version: str
    chunking_fingerprint: str
    collection: str
    vector_count: int
    created_at: datetime
    lineage: Lineage
    schema_version: int = SCHEMA_VERSION


@dataclasses.dataclass(frozen=True)
class Index:
    """One index: what one embedding fingerprint built and how a query must be embedded for it.

    Attributes:
        fingerprint: Fingerprint of the step: the identifier of the index.
        collection: Collection of the Search Index holding the points.
        model: The embedding model and its digest.
        dimensions: Length of every vector.
        distance: Similarity the collection is built for.
        chunking_fingerprint: Chunking fingerprint accepted; ``None`` when any.
        document_prefix: Put before a chunk's text when it was embedded.
        query_prefix: To put before a query when it is embedded against this index.
        truncate: Whether inputs beyond the context were cut.
        num_ctx: Context the model ran with, in tokens.
        created_at: When the index was first prepared, in UTC.
        image: Digest of the service image, when known.
        schema_version: Version of the stored record's schema.
    """

    fingerprint: str
    collection: str
    model: ModelRef
    dimensions: int
    distance: str
    chunking_fingerprint: str | None
    document_prefix: str
    query_prefix: str
    truncate: bool
    num_ctx: int
    created_at: datetime
    image: str | None = None
    schema_version: int = SCHEMA_VERSION


@dataclasses.dataclass(frozen=True)
class CollectionInfo:
    """What a Search Index backend reports about a collection.

    Attributes:
        dimensions: Length of the vectors.
        distance: Similarity, in the step's names.
        point_count: Points stored.
    """

    dimensions: int
    distance: str
    point_count: int
