"""The unit of chunking work: fetch one scene set's records, chunk them, store the result.

`scheduler` decides which scene sets a pass processes and how many at a time; the `Chunker`
decides what a chunk contains and is fingerprinted for it. This module only fetches, checks
liveness and completeness, stamps identity and lineage, and writes, so it sits outside the
fingerprint.
"""

import logging
from collections.abc import Iterable, Sequence

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import ChunkingConfig
from scene_rag.retrieval.chunking.protocols import (
    Chunker,
    ChunkStoreWriter,
    Clock,
    SceneRecordReader,
    VersionRetiredError,
)
from scene_rag.retrieval.chunking.report import Outcome

logger = logging.getLogger(__name__)


class ChunkingService:
    """`SceneSetChunker` over one `Chunker`.

    Chunks are written one by one, then the chunk set last: a worker that dies halfway leaves
    chunks that the rerun finds as duplicates, and the set stays pending until the marker is
    written. Before every write the asset version's liveness is checked, so a version retired
    meanwhile stops the work.
    """

    def __init__(
        self,
        *,
        reader: SceneRecordReader,
        writer: ChunkStoreWriter,
        chunker: Chunker,
        clock: Clock,
        config: ChunkingConfig,
    ) -> None:
        """Compose the unit of work from its collaborators; `bootstrap` is the only caller.

        Args:
            reader: Read view of the scene-api.
            writer: Command side of the Chunk Store.
            chunker: The chunking strategy.
            clock: Source of ``created_at``.
            config: Hyperparameters in force; the upstream pins and the hash algorithm.
        """
        self._reader = reader
        self._writer = writer
        self._chunker = chunker
        self._clock = clock
        self._config = config

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the chunker, stamped on every record."""
        return self._chunker.fingerprint

    def chunk(self, summary: models.SceneSetSummaryView) -> Outcome:
        """Chunk one scene set; see `SceneSetChunker.chunk`."""
        pins = models.resolve_pins(summary, self._config.upstream)
        if pins is None:
            logger.info("deferred %s: not complete under the required enrichers", summary.version)
            return "deferred"
        self._check_live(summary.asset_version)
        records = self._reader.list_scene_records(summary.version, pins)
        if not self._complete(summary, records, pins):
            return "deferred"
        if any(not record.live for record in records):
            raise VersionRetiredError(summary.asset_version)
        drafts = self._chunker.chunk(records)
        chunks = [self._chunk(summary, records, draft, index) for index, draft in enumerate(drafts)]
        self._check_live(summary.asset_version)
        written = 0
        for chunk in chunks:
            if self._writer.put_chunk(chunk):
                written += 1
            else:
                logger.info("chunk %d of %s already written", chunk.chunk_index, summary.version)
        set_version = models.chunk_set_version(
            summary.version, self.fingerprint, self._config.hash_algorithm
        )
        chunk_set = models.ChunkSet(
            version=set_version,
            scene_set_version=summary.version,
            asset_version=summary.asset_version,
            segmentation_fingerprint=summary.segmentation_fingerprint,
            pins=pins,
            chunk_count=len(chunks),
            created_at=self._clock.now(),
            lineage=self._lineage((record.version for record in records), set_version),
        )
        if not self._writer.put_chunk_set(chunk_set):
            logger.info("%s already chunked under %s", summary.version, self.fingerprint)
            return "duplicate"
        logger.info(
            "%s: %d chunk(s), %d written, %d scene(s)",
            summary.version,
            len(chunks),
            written,
            len(records),
        )
        return "chunked"

    def _check_live(self, asset_version: str) -> None:
        """Raise `VersionRetiredError` unless the asset version is still live."""
        current = self._reader.get_asset_version(asset_version)
        if current is None or not current.live:
            raise VersionRetiredError(asset_version)

    @staticmethod
    def _complete(
        summary: models.SceneSetSummaryView,
        records: Sequence[models.SceneRecordView],
        pins: Sequence[models.EnrichmentPin],
    ) -> bool:
        """Return whether the fetched records are the listed set, complete under the pins."""
        if [record.scene.index for record in records] != list(range(summary.scene_count)):
            logger.warning(
                "deferred %s: %d record(s) fetched for %d scene(s)",
                summary.version,
                len(records),
                summary.scene_count,
            )
            return False
        for record in records:
            missing = set(pins) - {item.pin for item in record.enrichments}
            if missing:
                logger.warning(
                    "deferred %s: scene %d lacks %s",
                    summary.version,
                    record.scene.index,
                    " ".join(str(pin) for pin in sorted(missing)),
                )
                return False
        return True

    def _lineage(self, inputs: Iterable[str], version: str) -> models.Lineage:
        return models.Lineage(
            fingerprint=self.fingerprint,
            inputs=tuple(models.SCENE_RECORD_INPUT_PREFIX + item for item in inputs),
            version=version,
        )

    def _chunk(
        self,
        summary: models.SceneSetSummaryView,
        records: Sequence[models.SceneRecordView],
        draft: models.ChunkDraft,
        index: int,
    ) -> models.Chunk:
        """Give a draft its identity, asset facts and provenance."""
        by_index = {record.scene.index: record for record in records}
        covered = [by_index[scene_index] for scene_index in draft.scene_indices]
        first = covered[0]
        versions = [record.version for record in covered]
        version = models.chunk_version(
            versions, self.fingerprint, index, self._config.hash_algorithm
        )
        classifications = {
            item.classification
            for record in covered
            for item in record.enrichments
            if item.ref in draft.enrichments
        }
        return models.Chunk(
            version=version,
            scene_set_version=summary.version,
            asset_version=summary.asset_version,
            segmentation_fingerprint=summary.segmentation_fingerprint,
            chunk_index=index,
            scene_indices=draft.scene_indices,
            start_seconds=draft.start_seconds,
            end_seconds=draft.end_seconds,
            text=draft.text,
            sources=first.asset.sources,
            entitlements=first.asset.entitlements,
            classifications=tuple(sorted(classifications)),
            keyframes=tuple(
                models.KeyframeRef(keyframe.derivative.digest, keyframe.seconds)
                for record in covered
                for keyframe in record.scene.keyframes
            ),
            enrichments=draft.enrichments,
            created_at=self._clock.now(),
            lineage=self._lineage(versions, version),
        )
