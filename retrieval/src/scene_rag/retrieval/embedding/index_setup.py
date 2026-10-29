"""Preparing the index of the current fingerprint, once per process, before the first pass.

Deployment code, outside the fingerprint: the collection is named after the fingerprint and
created or verified in the Search Index, and the index record is written once to the ledger
so query-api can find the collection and embed queries the way the index was built.
"""

import dataclasses
import logging

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.config import EmbeddingConfig
from scene_rag.retrieval.embedding.protocols import (
    Clock,
    EmbeddingModel,
    IndexMismatchError,
    IndexStoreReader,
    IndexStoreWriter,
    SearchIndexWriter,
)

logger = logging.getLogger(__name__)


def prepare_index(
    *,
    model: EmbeddingModel,
    index_writer: SearchIndexWriter,
    store_reader: IndexStoreReader,
    store_writer: IndexStoreWriter,
    config: EmbeddingConfig,
    clock: Clock,
) -> models.Index:
    """Ensure the collection and the index record of the model's fingerprint exist and agree.

    Args:
        model: The embedding model; its fingerprint names the index.
        index_writer: Command side of the Search Index.
        store_reader: Query side of the index ledger.
        store_writer: Command side of the index ledger.
        config: Hyperparameters in force.
        clock: Source of ``created_at``.

    Returns:
        The index record in force, as stored.

    Raises:
        IndexMismatchError: When the collection or the stored record was built differently.
        ModelError: When the model cannot report its dimensions.
        SearchIndexError: When the Search Index cannot be reached.
    """
    fingerprint = model.fingerprint
    collection = models.collection_name(fingerprint)
    dimensions = model.dimensions
    index_writer.ensure_collection(collection, dimensions, config.index.distance)
    index = models.Index(
        fingerprint=fingerprint,
        collection=collection,
        model=model.model,
        dimensions=dimensions,
        distance=config.index.distance,
        chunking_fingerprint=config.upstream.chunking_fingerprint,
        document_prefix=config.model.document_prefix,
        query_prefix=config.model.query_prefix,
        truncate=config.model.truncate,
        num_ctx=config.model.num_ctx,
        created_at=clock.now(),
    )
    if store_writer.put_index(index):
        logger.info("recorded index %s in collection %s", fingerprint, collection)
        return index
    stored = store_reader.get_index(fingerprint)
    if (
        stored is None
        or dataclasses.replace(stored, created_at=index.created_at, image=index.image) != index
    ):
        raise IndexMismatchError(
            f"index {fingerprint} is recorded with other values than the step produces: "
            f"{stored} vs {index}"
        )
    return stored
