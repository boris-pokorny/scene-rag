"""The unit of enrichment work: run the pending enrichers over the pending scenes of one set.

`scheduler` decides which scene sets a pass processes and how many at a time; each `Enricher`
decides what a record contains and is fingerprinted for it. This module only sequences the
calls, checks liveness and stamps lineage, so it sits outside every fingerprint.
"""

import logging
from collections.abc import Mapping, Sequence

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import EnrichmentConfig
from scene_rag.extraction.enrichment.protocols import (
    AssetVersionReader,
    Clock,
    DerivativeUnavailableError,
    Enricher,
    EnrichmentStoreWriter,
    SceneSetReader,
    VersionRetiredError,
)

logger = logging.getLogger(__name__)


class EnrichmentService:
    """`SceneSetEnricher` over a fixed sequence of enrichers.

    Records are written one scene at a time, as soon as each is produced, so a worker that dies
    halfway keeps what it had: the next pass derives the rest as pending. Before every write the
    asset version's liveness is checked, so a version retired meanwhile stops the work. A scene
    whose derivative cannot be read, or whose model call fails, is logged and skipped: the other
    scenes and enrichers of the set still get their records, and the set is reported deferred or
    failed once at the end, so one bad keyframe never blocks a whole set.
    """

    def __init__(
        self,
        *,
        assets: AssetVersionReader,
        scene_reader: SceneSetReader,
        writer: EnrichmentStoreWriter,
        enrichers: Sequence[Enricher],
        clock: Clock,
        config: EnrichmentConfig,
    ) -> None:
        """Compose the unit of work from its collaborators; `bootstrap` is the only caller.

        Args:
            assets: Read view of the Asset Registry, for the liveness checks.
            scene_reader: Read view of the Scene Store.
            writer: Command side of the Enrichment Store.
            enrichers: The enrichers to run, in order; their names must differ.
            clock: Source of ``created_at``.
            config: Hyperparameters in force; only the hash algorithm is used here.

        Raises:
            ValueError: When two enrichers share a name, or there is none.
        """
        names = [enricher.name for enricher in enrichers]
        if not names:
            raise ValueError("at least one enricher is required")
        if len(set(names)) != len(names):
            raise ValueError(f"enricher names must be unique: {names}")
        self._assets = assets
        self._scene_reader = scene_reader
        self._writer = writer
        self._enrichers = tuple(enrichers)
        self._clock = clock
        self._config = config

    @property
    def fingerprints(self) -> Mapping[str, str]:
        """Fingerprint per enricher name."""
        return {enricher.name: enricher.fingerprint for enricher in self._enrichers}

    def enrich(
        self, summary: models.SceneSetSummary, pending: Mapping[str, frozenset[int]]
    ) -> models.Outcome:
        """Enrich the pending scenes of one scene set; see `SceneSetEnricher.enrich`."""
        self._check_live(summary.asset_version)
        scene_set = self._scene_reader.get(summary.version)
        if scene_set is None:
            raise VersionRetiredError(f"scene set {summary.version} is gone")
        written = deferred = failed = 0
        for enricher in self._enrichers:
            for index in sorted(pending.get(enricher.name, ())):
                scene = self._scene_at(scene_set, index)
                if scene is None:
                    continue
                try:
                    written += self._enrich_scene(enricher, scene_set, scene)
                except DerivativeUnavailableError as exc:
                    deferred += 1
                    logger.warning(
                        "deferred %s of scene %d of %s: %s",
                        enricher.name,
                        scene.index,
                        scene_set.version,
                        exc,
                    )
                except VersionRetiredError:
                    raise
                except Exception:
                    failed += 1
                    logger.exception(
                        "%s of scene %d of %s failed", enricher.name, scene.index, scene_set.version
                    )
        logger.info(
            "%s: %d record(s) written, %d scene(s) deferred, %d failed",
            summary.version,
            written,
            deferred,
            failed,
        )
        if failed:
            return "failed"
        if deferred:
            return "deferred"
        return "enriched" if written else "duplicate"

    def _check_live(self, asset_version: str) -> None:
        """Raise `VersionRetiredError` unless the asset version is still live."""
        current = self._assets.get(asset_version)
        if current is None or not current.is_live:
            raise VersionRetiredError(asset_version)

    @staticmethod
    def _scene_at(scene_set: models.SceneSetView, index: int) -> models.SceneView | None:
        """Return the scene at ``index``, or ``None`` (logged) when the set holds no such scene."""
        if not 0 <= index < len(scene_set.scenes):
            logger.warning("scene %d not in %s; skipped", index, scene_set.version)
            return None
        scene = scene_set.scenes[index]
        if scene.index != index:
            logger.warning(
                "scene at position %d of %s is numbered %d; skipped",
                index,
                scene_set.version,
                scene.index,
            )
            return None
        return scene

    def _enrich_scene(
        self, enricher: Enricher, scene_set: models.SceneSetView, scene: models.SceneView
    ) -> int:
        """Run one enricher over one scene and store the record; return the records written."""
        output = enricher.enrich(scene_set, scene)
        self._check_live(scene_set.asset_version)
        version = models.enrichment_version(
            scene_set.version,
            scene.index,
            enricher.name,
            enricher.fingerprint,
            self._config.hash_algorithm,
        )
        record = models.Enrichment(
            version=version,
            scene_set_version=scene_set.version,
            asset_version=scene_set.asset_version,
            scene_index=scene.index,
            enricher=enricher.name,
            classification=enricher.classification,
            model=enricher.model,
            segments=output.segments,
            created_at=self._clock.now(),
            lineage=models.Lineage(
                fingerprint=enricher.fingerprint,
                inputs=(
                    models.SCENE_SET_INPUT_PREFIX + scene_set.version,
                    *(models.DERIVATIVE_INPUT_PREFIX + digest for digest in output.inputs),
                ),
                version=version,
            ),
        )
        if not self._writer.put(record):
            logger.info(
                "%s of scene %d of %s already written",
                enricher.name,
                scene.index,
                scene_set.version,
            )
            return 0
        logger.info(
            "%s of scene %d of %s: %d segment(s)",
            enricher.name,
            scene.index,
            scene_set.version,
            len(output.segments),
        )
        return 1
