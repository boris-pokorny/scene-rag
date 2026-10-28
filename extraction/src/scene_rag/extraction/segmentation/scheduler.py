"""One segmentation pass: derive the pending versions and run the segmenter over them.

Orchestration only, deliberately outside the fingerprint: how many versions are in flight,
and in which order, changes nothing about what a scene set contains.
"""

import logging

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.protocols import (
    AssetVersionReader,
    Dispatcher,
    MediaUnavailableError,
    SceneStoreReader,
    VersionRetiredError,
    VersionSegmenter,
)

logger = logging.getLogger(__name__)


class Scheduler:
    """Segments every live asset version that has no scene set under the current fingerprint.

    Pending work is derived, never flagged: a pass lists the live versions and the asset
    versions already segmented under the segmenter's fingerprint, and hands the difference to
    the dispatcher. Each version's outcome is counted; a failing version is logged and skipped,
    a failing listing propagates, since nothing can be decided without it.
    """

    def __init__(
        self,
        *,
        assets: AssetVersionReader,
        scene_reader: SceneStoreReader,
        segmenter: VersionSegmenter,
        dispatcher: Dispatcher,
    ) -> None:
        """Compose the pass; `bootstrap` is the only caller.

        Args:
            assets: Read view of the Asset Registry.
            scene_reader: Query side of the Scene Store.
            segmenter: Unit of work per version.
            dispatcher: Concurrency policy the versions of a pass run under.
        """
        self._assets = assets
        self._scene_reader = scene_reader
        self._segmenter = segmenter
        self._dispatcher = dispatcher

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the segmenter driven by this pass."""
        return self._segmenter.fingerprint

    def run_once(self) -> models.SegmentationReport:
        """Run one pass over the pending versions.

        Returns:
            Counters of what the pass did.
        """
        fingerprint = self._segmenter.fingerprint
        done = self._scene_reader.list_asset_versions(fingerprint)
        pending = [record for record in self._assets.list_live() if record.version not in done]
        logger.info("%d asset version(s) pending under fingerprint %s", len(pending), fingerprint)
        report = models.SegmentationReport.counting(self._dispatcher.map(self._outcome_of, pending))
        logger.info("segmentation pass: %s", report)
        return report

    def _outcome_of(self, record: models.AssetVersionView) -> models.Outcome:
        try:
            return self._segmenter.segment(record)
        except MediaUnavailableError as exc:
            logger.warning("deferred %s: %s", record.version, exc)
            return "deferred"
        except VersionRetiredError:
            logger.info("dropped %s: retired while segmenting", record.version)
            return "dropped"
        except Exception:
            logger.exception("failed to segment %s", record.version)
            return "failed"
