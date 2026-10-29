"""Index ledger backend over MongoDB.

`MongoIndexStore` implements `IndexStoreReader` and `IndexStoreWriter` over two collections
in the Chunk Store's database: ``index_sets`` holds the completion marker of each chunk set
under each embedding fingerprint, ``indexes`` one record per fingerprint saying what was
built and how a query must be embedded for it. A record is inserted once and never updated.
"""

from datetime import UTC, datetime
from typing import Any, Final

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from scene_rag.retrieval.embedding import models

INDEX_SETS_COLLECTION: Final = "index_sets"
INDEXES_COLLECTION: Final = "indexes"

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


def index_set_to_document(index_set: models.IndexSet) -> Document:
    """Encode an index set as a store document (``contracts/internal/index-set.schema.json``)."""
    return {
        "_id": index_set.version,
        "schema_version": index_set.schema_version,
        "chunk_set_version": index_set.chunk_set_version,
        "scene_set_version": index_set.scene_set_version,
        "asset_version": index_set.asset_version,
        "chunking_fingerprint": index_set.chunking_fingerprint,
        "collection": index_set.collection,
        "vector_count": index_set.vector_count,
        "created_at": index_set.created_at,
        "lineage": lineage_to_document(index_set.lineage),
    }


def index_set_from_document(document: Document) -> models.IndexSet:
    """Decode a store document into an index set."""
    return models.IndexSet(
        version=document["_id"],
        chunk_set_version=document["chunk_set_version"],
        scene_set_version=document["scene_set_version"],
        asset_version=document["asset_version"],
        chunking_fingerprint=document["chunking_fingerprint"],
        collection=document["collection"],
        vector_count=document["vector_count"],
        created_at=_utc(document["created_at"]),
        lineage=lineage_from_document(document["lineage"]),
        schema_version=document["schema_version"],
    )


def index_to_document(index: models.Index) -> Document:
    """Encode an index record as a store document (``contracts/internal/index.schema.json``)."""
    return {
        "_id": index.fingerprint,
        "schema_version": index.schema_version,
        "collection": index.collection,
        "model": {"name": index.model.name, "digest": index.model.digest},
        "dimensions": index.dimensions,
        "distance": index.distance,
        "chunking_fingerprint": index.chunking_fingerprint,
        "document_prefix": index.document_prefix,
        "query_prefix": index.query_prefix,
        "truncate": index.truncate,
        "num_ctx": index.num_ctx,
        "created_at": index.created_at,
        "image": index.image,
    }


def index_from_document(document: Document) -> models.Index:
    """Decode a store document into an index record."""
    return models.Index(
        fingerprint=document["_id"],
        collection=document["collection"],
        model=models.ModelRef(document["model"]["name"], document["model"]["digest"]),
        dimensions=document["dimensions"],
        distance=document["distance"],
        chunking_fingerprint=document["chunking_fingerprint"],
        document_prefix=document["document_prefix"],
        query_prefix=document["query_prefix"],
        truncate=document["truncate"],
        num_ctx=document["num_ctx"],
        created_at=_utc(document["created_at"]),
        image=document["image"],
        schema_version=document["schema_version"],
    )


class MongoIndexStore:
    """Index ledger over two MongoDB collections, ``_id`` being each record's identifier."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to ``database`` on ``client`` and ensure the indexes exist.

        ``(lineage.fingerprint, chunk_set_version)`` serves `list_done`; ``asset_version``
        serves the purge of retired assets.
        """
        self._index_sets = client[database][INDEX_SETS_COLLECTION]
        self._indexes = client[database][INDEXES_COLLECTION]
        self._index_sets.create_index(
            [("lineage.fingerprint", ASCENDING), ("chunk_set_version", ASCENDING)]
        )
        self._index_sets.create_index([("asset_version", ASCENDING)])

    def get_index_set(self, version: str) -> models.IndexSet | None:
        """Return the index set with the given data version."""
        document = self._index_sets.find_one({"_id": version})
        return None if document is None else index_set_from_document(document)

    def get_index(self, fingerprint: str) -> models.Index | None:
        """Return the index record of the fingerprint."""
        document = self._indexes.find_one({"_id": fingerprint})
        return None if document is None else index_from_document(document)

    def list_done(self, fingerprint: str) -> frozenset[str]:
        """Return the chunk set versions with an index set under the fingerprint."""
        cursor = self._index_sets.find(
            {"lineage.fingerprint": fingerprint},
            projection={"_id": False, "chunk_set_version": True},
        )
        return frozenset(document["chunk_set_version"] for document in cursor)

    def put_index_set(self, index_set: models.IndexSet) -> bool:
        """Insert the index set; ``False`` when its version already exists."""
        try:
            self._index_sets.insert_one(index_set_to_document(index_set))
        except DuplicateKeyError:
            return False
        return True

    def put_index(self, index: models.Index) -> bool:
        """Insert the index record; ``False`` when its fingerprint already has one."""
        try:
            self._indexes.insert_one(index_to_document(index))
        except DuplicateKeyError:
            return False
        return True
