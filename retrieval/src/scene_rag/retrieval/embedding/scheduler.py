"""One embedding pass: derive the pending chunk sets and run the embedder over them.

Orchestration only, deliberately outside the fingerprint: how many chunk sets are in flight,
and in which order, changes nothing about what a vector contains.
"""

import logging

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.config import EmbeddingConfig
from scene_rag.retrieval.embedding.protocols import (
    ChunkReader,
    ChunkSetEmbedder,
    Dispatcher,
    IndexStoreReader,
)
from scene_rag.retrieval.embedding.report import EmbeddingReport, Outcome

logger = logging.getLogger(__name__)


class Scheduler:
    """Embeds every chunk set lacking an index set under the current fingerprint.

    Pending work is derived, never flagged: a pass lists the chunk sets under the accepted
    chunking fingerprint, subtracts those with an index set under the step's fingerprint,
    and hands the rest to the dispatcher, one chunk set at a time. A failing listing
    propagates, since nothing can be decided without it.
    """

    def __init__(
        self,
        *,
        chunk_reader: ChunkReader,
        store_reader: IndexStoreReader,
        embedder: ChunkSetEmbedder,
        dispatcher: Dispatcher,
        config: EmbeddingConfig,
    ) -> None:
        """Compose the pass; `bootstrap` is the only caller.

        Args:
            chunk_reader: Read view of the Chunk Store.
            store_reader: Query side of the index ledger.
            embedder: Unit of work per chunk set.
            dispatcher: Concurrency policy the chunk sets of a pass run under.
            config: Hyperparameters in force; the upstream section selects the chunk sets.
        """
        self._chunk_reader = chunk_reader
        self._store_reader = store_reader
        self._embedder = embedder
        self._dispatcher = dispatcher
        self._upstream = config.upstream

    @property
    def fingerprint(self) -> str:
        """Fingerprint driven by this pass."""
        return self._embedder.fingerprint

    def run_once(self) -> EmbeddingReport:
        """Run one pass over the pending chunk sets.

        Returns:
            Counters of what the pass did, per chunk set.
        """
        done = self._store_reader.list_done(self.fingerprint)
        work = [
            chunk_set
            for chunk_set in self._chunk_reader.list_chunk_sets(self._upstream.chunking_fingerprint)
            if chunk_set.version not in done
        ]
        logger.info("%d chunk set(s) pending under %s", len(work), self.fingerprint)
        report = EmbeddingReport.counting(self._dispatcher.map(self._outcome_of, work))
        logger.info("embedding pass: %s", report)
        return report

    def _outcome_of(self, chunk_set: models.ChunkSetView) -> Outcome:
        try:
            return self._embedder.embed(chunk_set)
        except Exception:
            logger.exception("failed to embed %s", chunk_set.version)
            return "failed"
