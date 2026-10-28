"""The per-scene chunker: one retrieval unit per scene, from the text of its enrichments."""

import logging
from collections.abc import Sequence
from typing import Final

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import ChunkingConfig
from scene_rag.retrieval.chunking.fingerprint import compute_fingerprint

logger = logging.getLogger(__name__)

NAME: Final = "per_scene"
# The modules that decide what this chunker writes; see `fingerprint.compute_fingerprint`.
FINGERPRINTED_MODULES: Final = ("models.py", "fingerprint.py", "per_scene.py")


class PerSceneChunker:
    """`Chunker` keeping scene boundaries: a chunk is one scene's text.

    Per enricher in the configured order, the segments of that enricher are sorted by start,
    stripped and joined; the non-empty sections are joined in turn. A scene without text gets
    no chunk. The records carry one fingerprint per enricher (the service fetches them pinned),
    so grouping by enricher name never mixes two fingerprints.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        """Compose the chunker; `bootstrap` is the only caller.

        Args:
            config: Hyperparameters in force.
        """
        self._config = config.per_scene
        self._fingerprint = compute_fingerprint(config, FINGERPRINTED_MODULES)

    @property
    def name(self) -> str:
        """Return ``"per_scene"``."""
        return NAME

    @property
    def fingerprint(self) -> str:
        """Return the chunker's fingerprint."""
        return self._fingerprint

    def chunk(self, records: Sequence[models.SceneRecordView]) -> tuple[models.ChunkDraft, ...]:
        """Draft one chunk per record with text; see `Chunker.chunk`."""
        drafts: list[models.ChunkDraft] = []
        for record in records:
            sections: list[str] = []
            used: list[models.EnrichmentRef] = []
            for enricher in self._config.enrichers:
                matching = [item for item in record.enrichments if item.enricher == enricher]
                segments = sorted(
                    (segment for item in matching for segment in item.segments),
                    key=lambda segment: (segment.start_seconds, segment.end_seconds),
                )
                text = self._config.segment_separator.join(
                    stripped for segment in segments if (stripped := segment.text.strip())
                )
                if not text:
                    continue
                sections.append(text)
                used.extend(
                    item.ref for item in matching if any(s.text.strip() for s in item.segments)
                )
            if not sections:
                logger.info(
                    "scene %d of %s has no text", record.scene.index, record.scene_set_version
                )
                continue
            drafts.append(
                models.ChunkDraft(
                    scene_indices=(record.scene.index,),
                    start_seconds=record.scene.start_seconds,
                    end_seconds=record.scene.end_seconds,
                    text=self._config.section_separator.join(sections),
                    enrichments=tuple(used),
                )
            )
        return tuple(drafts)
