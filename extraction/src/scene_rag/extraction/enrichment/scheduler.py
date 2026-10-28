"""One enrichment pass: derive the pending scenes and run the enricher over their scene sets.

Orchestration only, deliberately outside every fingerprint: how many scene sets are in flight,
and in which order, changes nothing about what a record contains.
"""

import logging
from collections.abc import Mapping

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.protocols import (
    AssetVersionReader,
    Dispatcher,
    EnrichmentStoreReader,
    SceneSetEnricher,
    SceneSetReader,
    VersionRetiredError,
)

logger = logging.getLogger(__name__)

type Work = tuple[models.SceneSetSummary, Mapping[str, frozenset[int]]]
"""A scene set and, per enricher name, the scenes still to enrich."""


class Scheduler:
    """Enriches every scene of a live asset version lacking a record under the current fingerprint.

    Pending work is derived, never flagged: a pass lists the live asset versions, the scene sets
    and, per enricher, the scenes already enriched under its fingerprint, and hands the
    difference to the dispatcher, one scene set at a time. Every scene set of a live version is
    enriched, whichever segmentation fingerprint cut it: old and new coexist until a release
    gate decides. The unit of work reports a scene set that lost a scene to a missing derivative
    or a failing backend; a scene set that cannot be loaded at all is logged and skipped here; a
    failing listing propagates, since nothing can be decided without it.
    """

    def __init__(
        self,
        *,
        assets: AssetVersionReader,
        scene_reader: SceneSetReader,
        enrichment_reader: EnrichmentStoreReader,
        enricher: SceneSetEnricher,
        dispatcher: Dispatcher,
    ) -> None:
        """Compose the pass; `bootstrap` is the only caller.

        Args:
            assets: Read view of the Asset Registry.
            scene_reader: Read view of the Scene Store.
            enrichment_reader: Query side of the Enrichment Store.
            enricher: Unit of work per scene set.
            dispatcher: Concurrency policy the scene sets of a pass run under.
        """
        self._assets = assets
        self._scene_reader = scene_reader
        self._enrichment_reader = enrichment_reader
        self._enricher = enricher
        self._dispatcher = dispatcher

    @property
    def fingerprints(self) -> Mapping[str, str]:
        """Fingerprint per enricher name, as driven by this pass."""
        return self._enricher.fingerprints

    def run_once(self) -> models.EnrichmentReport:
        """Run one pass over the pending scenes.

        Returns:
            Counters of what the pass did, per scene set.
        """
        live = self._assets.list_live_versions()
        done = {
            name: self._enrichment_reader.list_done(name, fingerprint)
            for name, fingerprint in self.fingerprints.items()
        }
        work: list[Work] = []
        scenes_pending = 0
        for summary in self._scene_reader.list_summaries():
            if summary.asset_version not in live:
                continue
            pending = {
                name: missing
                for name, finished in done.items()
                if (
                    missing := frozenset(
                        index
                        for index in range(summary.scene_count)
                        if (summary.version, index) not in finished
                    )
                )
            }
            if pending:
                work.append((summary, pending))
                scenes_pending += sum(len(indices) for indices in pending.values())
        logger.info(
            "%d scene set(s), %d scene enrichment(s) pending under %s",
            len(work),
            scenes_pending,
            " ".join(f"{name}={fp}" for name, fp in self.fingerprints.items()),
        )
        report = models.EnrichmentReport.counting(self._dispatcher.map(self._outcome_of, work))
        logger.info("enrichment pass: %s", report)
        return report

    def _outcome_of(self, work: Work) -> models.Outcome:
        summary, pending = work
        try:
            return self._enricher.enrich(summary, pending)
        except VersionRetiredError as exc:
            logger.info("dropped %s: %s", summary.version, exc)
            return "dropped"
        except Exception:
            logger.exception("failed to enrich %s", summary.version)
            return "failed"
