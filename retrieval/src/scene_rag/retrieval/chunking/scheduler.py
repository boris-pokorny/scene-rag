"""One chunking pass: derive the pending scene sets and run the chunker over them.

Orchestration only, deliberately outside the fingerprint: how many scene sets are in flight,
and in which order, changes nothing about what a chunk contains.
"""

import logging

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import ChunkingConfig
from scene_rag.retrieval.chunking.protocols import (
    ChunkStoreReader,
    Dispatcher,
    SceneRecordReader,
    SceneSetChunker,
    VersionRetiredError,
)
from scene_rag.retrieval.chunking.report import ChunkingReport, Outcome

logger = logging.getLogger(__name__)


class Scheduler:
    """Chunks every listed scene set lacking a chunk set under the current fingerprint.

    Pending work is derived, never flagged: a pass asks scene-api for the scene sets of live
    asset versions under the accepted upstream fingerprints, subtracts those with a chunk set
    under the step's fingerprint, drops those not yet complete under the required enrichers,
    and hands the rest to the dispatcher, one scene set at a time. A failing listing
    propagates, since nothing can be decided without it.
    """

    def __init__(
        self,
        *,
        reader: SceneRecordReader,
        chunk_reader: ChunkStoreReader,
        chunker: SceneSetChunker,
        dispatcher: Dispatcher,
        config: ChunkingConfig,
    ) -> None:
        """Compose the pass; `bootstrap` is the only caller.

        Args:
            reader: Read view of the scene-api.
            chunk_reader: Query side of the Chunk Store.
            chunker: Unit of work per scene set.
            dispatcher: Concurrency policy the scene sets of a pass run under.
            config: Hyperparameters in force; the upstream section selects the scene sets.
        """
        self._reader = reader
        self._chunk_reader = chunk_reader
        self._chunker = chunker
        self._dispatcher = dispatcher
        self._upstream = config.upstream

    @property
    def fingerprint(self) -> str:
        """Fingerprint driven by this pass."""
        return self._chunker.fingerprint

    def run_once(self) -> ChunkingReport:
        """Run one pass over the pending scene sets.

        Returns:
            Counters of what the pass did, per scene set.
        """
        done = self._chunk_reader.list_done(self.fingerprint)
        query = models.SceneSetQuery(
            segmentation_fingerprint=self._upstream.segmentation_fingerprint,
            pins=tuple(models.EnrichmentPin.parse(pin) for pin in self._upstream.enrichment_pins),
        )
        work: list[models.SceneSetSummaryView] = []
        not_ready = 0
        for summary in self._reader.list_scene_sets(query):
            if summary.version in done or not summary.live:
                continue
            if models.resolve_pins(summary, self._upstream) is None:
                not_ready += 1
                continue
            work.append(summary)
        logger.info(
            "%d scene set(s) pending, %d not yet complete, under %s",
            len(work),
            not_ready,
            self.fingerprint,
        )
        report = ChunkingReport.counting(self._dispatcher.map(self._outcome_of, work))
        logger.info("chunking pass: %s", report)
        return report

    def _outcome_of(self, summary: models.SceneSetSummaryView) -> Outcome:
        try:
            return self._chunker.chunk(summary)
        except VersionRetiredError as exc:
            logger.info("dropped %s: %s", summary.version, exc)
            return "dropped"
        except Exception:
            logger.exception("failed to chunk %s", summary.version)
            return "failed"
