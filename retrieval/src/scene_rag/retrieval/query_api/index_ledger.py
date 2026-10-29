"""Read model over the index ledger the embedding step writes.

`MongoIndexLedger` is the role's `IndexLedgerReader` over the ``indexes`` collection in the
Chunk Store's database. The decoder is this role's copy of the internal contract
(``contracts/internal/index.schema.json``); ``tests/contract/`` pins it to the sample. The
role never writes: no index is created here, nothing is inserted.
"""

from datetime import UTC, datetime
from typing import Any, Final

from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.protocols import IndexUnavailableError

INDEXES_COLLECTION: Final = "indexes"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]


def _utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime; pymongo returns naive values unless told otherwise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def index_from_document(document: Document) -> models.IndexView:
    """Decode a ledger document as ``index.schema.json`` describes it.

    Raises:
        ValueError: When the document's schema version is not the one this role decodes.
        KeyError: When a required field is missing.
    """
    version = document["schema_version"]
    if version != models.INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"index record has schema_version {version!r}; this role decodes "
            f"{models.INDEX_SCHEMA_VERSION}"
        )
    return models.IndexView(
        fingerprint=document["_id"],
        collection=document["collection"],
        model=models.ModelRef(document["model"]["name"], document["model"]["digest"]),
        dimensions=document["dimensions"],
        distance=document["distance"],
        chunking_fingerprint=document["chunking_fingerprint"],
        query_prefix=document["query_prefix"],
        truncate=document["truncate"],
        num_ctx=document["num_ctx"],
        created_at=_utc(document["created_at"]),
    )


class MongoIndexLedger:
    """`IndexLedgerReader` over the ``indexes`` collection."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to ``database`` on ``client``; nothing is created."""
        self._indexes = client[database][INDEXES_COLLECTION]

    def _decode(self, document: Document | None) -> models.IndexView | None:
        if document is None:
            return None
        try:
            return index_from_document(document)
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexUnavailableError(
                f"index record {document.get('_id')!r} cannot be decoded: {exc}"
            ) from exc

    def get_index(self, fingerprint: str) -> models.IndexView | None:
        """Return the index built under the fingerprint; see `IndexLedgerReader.get_index`."""
        try:
            return self._decode(self._indexes.find_one({"_id": fingerprint}))
        except PyMongoError as exc:
            raise IndexUnavailableError(f"the index ledger cannot be read: {exc}") from exc

    def newest_index(self) -> models.IndexView | None:
        """Return the most recently prepared index; see `IndexLedgerReader.newest_index`."""
        try:
            document = self._indexes.find_one(
                sort=[("created_at", DESCENDING), ("_id", DESCENDING)]
            )
        except PyMongoError as exc:
            raise IndexUnavailableError(f"the index ledger cannot be read: {exc}") from exc
        return self._decode(document)
