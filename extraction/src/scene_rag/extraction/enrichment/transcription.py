"""The speech transcription enricher: what is said in a scene, from its audio clip."""

import logging
from typing import Final

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import TranscriptionConfig
from scene_rag.extraction.enrichment.fingerprint import compute_fingerprint
from scene_rag.extraction.enrichment.protocols import (
    DerivativeReader,
    Transcriber,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

NAME: Final = "transcript"
CLASSIFICATION: Final = "internal"
# The modules that decide what this enricher writes; see `fingerprint.compute_fingerprint`.
FINGERPRINTED_MODULES: Final = (
    "models.py",
    "fingerprint.py",
    "transcription.py",
    "whisper_backend.py",
)


class TranscriptionEnricher:
    """`Enricher` transcribing each scene's audio clip into timed segments.

    Segment times are offset from the clip to the stream and clamped to the scene's span; a
    scene without a clip, or without speech, gets no segments.
    """

    def __init__(
        self,
        *,
        transcriber: Transcriber,
        derivatives: DerivativeReader,
        config: TranscriptionConfig,
    ) -> None:
        """Compose the enricher; `bootstrap` is the only caller.

        Args:
            transcriber: The speech model.
            derivatives: Reader of the clip bytes.
            config: Hyperparameters in force.
        """
        self._transcriber = transcriber
        self._derivatives = derivatives
        self._config = config
        self._fingerprint = compute_fingerprint(
            config, FINGERPRINTED_MODULES, transcriber.model.digest
        )

    @property
    def name(self) -> str:
        """Return ``"transcript"``."""
        return NAME

    @property
    def classification(self) -> str:
        """Return the data classification of the transcripts."""
        return CLASSIFICATION

    @property
    def model(self) -> models.ModelRef:
        """Return the speech model and its digest."""
        return self._transcriber.model

    @property
    def fingerprint(self) -> str:
        """Return the enricher's fingerprint."""
        return self._fingerprint

    def enrich(
        self, scene_set: models.SceneSetView, scene: models.SceneView
    ) -> models.EnricherOutput:
        """Transcribe one scene; see `Enricher.enrich`."""
        clip = scene.audio_clip
        if clip is None:
            logger.info("scene %d of %s has no audio clip", scene.index, scene_set.version)
            return models.EnricherOutput()
        transcript = self._transcriber.transcribe(self._derivatives.read(clip))
        segments = tuple(
            segment
            for heard in transcript.segments
            if (segment := _placed(heard, scene, transcript.language)) is not None
        )
        logger.info(
            "transcribed %d segment(s) for scene %d of %s",
            len(segments),
            scene.index,
            scene_set.version,
        )
        return models.EnricherOutput(segments=segments, inputs=(clip.digest,))


def _placed(
    heard: TranscriptSegment, scene: models.SceneView, language: str | None
) -> models.SemanticSegment | None:
    """Move a clip-relative span into the stream, clamped to the scene; ``None`` when empty."""
    text = heard.text.strip()
    if not text:
        return None
    start = min(scene.start_seconds + heard.start_seconds, scene.end_seconds)
    end = max(start, min(scene.start_seconds + heard.end_seconds, scene.end_seconds))
    return models.SemanticSegment(
        start_seconds=start,
        end_seconds=end,
        text=text,
        language=language,
        confidence=heard.confidence,
    )
