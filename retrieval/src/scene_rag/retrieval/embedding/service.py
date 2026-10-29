"""The unit of embedding work: read one chunk set's chunks, embed them, write points and marker.

`scheduler` decides which chunk sets a pass processes and how many at a time; the
`EmbeddingModel` decides what a vector is and is fingerprinted for it. This module only
fetches, checks completeness, stamps identity and lineage, batches and writes, so it sits
outside the fingerprint.
"""

import logging
from collections.abc import Sequence

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.config import EmbeddingConfig
from scene_rag.retrieval.embedding.protocols import (
    ChunkReader,
    Clock,
    EmbeddingModel,
    IndexStoreWriter,
    ModelError,
    SearchIndexWriter,
)
from scene_rag.retrieval.embedding.report import Outcome

logger = logging.getLogger(__name__)


class EmbeddingService:
    """`ChunkSetEmbedder` over one `EmbeddingModel` and one index.

    Points are upserted batch by batch, then the index set last: a worker that dies halfway
    leaves points the rerun overwrites with the same content, and the chunk set stays pending
    until the marker is written.
    """

    def __init__(
        self,
        *,
        reader: ChunkReader,
        model: EmbeddingModel,
        index_writer: SearchIndexWriter,
        store_writer: IndexStoreWriter,
        index: models.Index,
        clock: Clock,
        config: EmbeddingConfig,
        batch_size: int,
    ) -> None:
        """Compose the unit of work from its collaborators; `bootstrap` is the only caller.

        Args:
            reader: Read view of the Chunk Store.
            model: The embedding model.
            index_writer: Command side of the Search Index.
            store_writer: Command side of the index ledger.
            index: The index in force, from `index_setup.prepare_index`.
            clock: Source of ``created_at``.
            config: Hyperparameters in force; the hash algorithm.
            batch_size: Texts embedded per request.

        Raises:
            ValueError: When the batch size is below one or the index is not the model's.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if index.fingerprint != model.fingerprint:
            raise ValueError(
                f"index {index.fingerprint} is not the model's fingerprint {model.fingerprint}"
            )
        self._reader = reader
        self._model = model
        self._index_writer = index_writer
        self._store_writer = store_writer
        self._index = index
        self._clock = clock
        self._config = config
        self._batch_size = batch_size

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the model, stamped on every record."""
        return self._model.fingerprint

    def embed(self, chunk_set: models.ChunkSetView) -> Outcome:
        """Embed one chunk set; see `ChunkSetEmbedder.embed`."""
        chunks = self._reader.list_chunks(
            chunk_set.scene_set_version, chunk_set.chunking_fingerprint
        )
        if [chunk.chunk_index for chunk in chunks] != list(range(chunk_set.chunk_count)):
            logger.warning(
                "deferred %s: %d chunk(s) fetched for %d listed",
                chunk_set.version,
                len(chunks),
                chunk_set.chunk_count,
            )
            return "deferred"
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._model.embed([chunk.text for chunk in batch])
            self._check(vectors, batch)
            self._index_writer.upsert(
                self._index.collection,
                [
                    models.Point(
                        id=models.point_id(chunk.version),
                        vector=vector,
                        entry=self._entry(chunk_set, chunk),
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
            )
        version = models.index_set_version(
            chunk_set.version, self.fingerprint, self._config.hash_algorithm
        )
        index_set = models.IndexSet(
            version=version,
            chunk_set_version=chunk_set.version,
            scene_set_version=chunk_set.scene_set_version,
            asset_version=chunk_set.asset_version,
            chunking_fingerprint=chunk_set.chunking_fingerprint,
            collection=self._index.collection,
            vector_count=len(chunks),
            created_at=self._clock.now(),
            lineage=self._lineage([chunk.version for chunk in chunks], version),
        )
        if not self._store_writer.put_index_set(index_set):
            logger.info("%s already embedded under %s", chunk_set.version, self.fingerprint)
            return "duplicate"
        logger.info(
            "%s: %d vector(s) in %s", chunk_set.version, len(chunks), self._index.collection
        )
        return "indexed"

    def _check(self, vectors: Sequence[models.Vector], batch: Sequence[models.ChunkView]) -> None:
        """Raise `ModelError` unless there is one vector of the index's length per chunk."""
        if len(vectors) != len(batch):
            raise ModelError(f"{len(vectors)} vector(s) for {len(batch)} chunk(s)")
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != self._index.dimensions:
                raise ModelError(
                    f"chunk {chunk.version} embedded to {len(vector)} dimensions; "
                    f"the index holds {self._index.dimensions}"
                )

    def _lineage(self, inputs: Sequence[str], version: str) -> models.Lineage:
        return models.Lineage(
            fingerprint=self.fingerprint,
            inputs=tuple(models.CHUNK_INPUT_PREFIX + item for item in inputs),
            version=version,
        )

    def _entry(self, chunk_set: models.ChunkSetView, chunk: models.ChunkView) -> models.IndexEntry:
        """Give a chunk's payload its identity and provenance."""
        version = models.index_entry_version(
            chunk.version, self.fingerprint, self._config.hash_algorithm
        )
        return models.IndexEntry(
            version=version,
            chunk_version=chunk.version,
            chunk_set_version=chunk_set.version,
            scene_set_version=chunk.scene_set_version,
            asset_version=chunk.asset_version,
            chunking_fingerprint=chunk.chunking_fingerprint,
            segmentation_fingerprint=chunk.segmentation_fingerprint,
            chunk_index=chunk.chunk_index,
            scene_indices=chunk.scene_indices,
            start_seconds=chunk.start_seconds,
            end_seconds=chunk.end_seconds,
            text=chunk.text,
            sources=chunk.sources,
            entitlements=chunk.entitlements,
            classifications=chunk.classifications,
            created_at=self._clock.now(),
            lineage=self._lineage([chunk.version], version),
        )
