"""Speech model over a loaded faster-whisper (CTranslate2) model.

This module is fingerprinted, so it holds only what decides the transcript: how a clip is
decoded and which options the model is called with. Loading the model, and hashing its
snapshot, is `whisper_loader`'s job, outside the fingerprint. Clips are decoded here from PCM
WAV into the 16 kHz float samples the model consumes, with the standard library rather than the
library's own media decoder: the clips are rendered in exactly that form by segmentation, and
the decoder path stays independent of PyAV's API.
"""

import io
import logging
import math
import wave
from collections.abc import Iterable
from typing import Any, Final, Protocol

import numpy as np

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import TranscriptionConfig
from scene_rag.extraction.enrichment.protocols import ModelError, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

# What Whisper models are trained on, and what faster-whisper assumes of a sample array.
SAMPLE_RATE: Final = 16000
_PCM16_BYTES: Final = 2
_PCM16_SCALE: Final = 32768.0

type Samples = np.ndarray[tuple[int], np.dtype[np.float32]]
"""Mono samples at `SAMPLE_RATE`, in ``[-1, 1]``."""


class WhisperError(ModelError):
    """The speech model could not be loaded or could not transcribe a clip."""


class WhisperLike(Protocol):
    """The part of ``faster_whisper.WhisperModel`` the backend uses."""

    def transcribe(self, audio: Samples, **options: Any) -> tuple[Iterable[Any], Any]:  # noqa: ANN401
        """Return the segments (lazily decoded) and the transcription info."""
        ...


def decode_pcm_wave(audio: bytes) -> Samples:
    """Decode a 16-bit PCM WAV clip at `SAMPLE_RATE` into mono float samples.

    Several channels are averaged into one.

    Raises:
        WhisperError: When the bytes are not such a clip; segmentation renders clips at that
            rate and width, so anything else is a configuration mismatch, not a bad asset.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as clip:
            rate, width, channels = clip.getframerate(), clip.getsampwidth(), clip.getnchannels()
            frames = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError) as exc:
        raise WhisperError(f"audio clip is not a WAV file: {exc}") from exc
    if rate != SAMPLE_RATE or width != _PCM16_BYTES:
        raise WhisperError(
            f"audio clip is {rate} Hz {8 * width}-bit; the speech model needs "
            f"{SAMPLE_RATE} Hz 16-bit PCM, which segmentation renders by default"
        )
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / _PCM16_SCALE
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


class FasterWhisperTranscriber:
    """`Transcriber` over a loaded faster-whisper model."""

    def __init__(
        self, model: WhisperLike, model_ref: models.ModelRef, config: TranscriptionConfig
    ) -> None:
        """Wrap a loaded model.

        Args:
            model: The loaded model.
            model_ref: Its name and snapshot digest.
            config: Hyperparameters; the decoding options.
        """
        self._model = model
        self._model_ref = model_ref
        self._options: dict[str, object] = {
            "language": config.language,
            "beam_size": config.beam_size,
            "temperature": config.temperature,
            "vad_filter": config.vad_filter,
            "condition_on_previous_text": config.condition_on_previous_text,
        }

    @property
    def model(self) -> models.ModelRef:
        """Return the model and its digest."""
        return self._model_ref

    def transcribe(self, audio: bytes) -> Transcript:
        """Transcribe one clip; see `Transcriber.transcribe`. An empty clip has no segments."""
        samples = decode_pcm_wave(audio)
        if samples.size == 0:
            return Transcript(language=None, segments=())
        try:
            segments, info = self._model.transcribe(samples, **self._options)
            heard = tuple(
                TranscriptSegment(
                    start_seconds=float(segment.start),
                    end_seconds=float(segment.end),
                    text=str(segment.text),
                    confidence=_confidence(segment.avg_logprob),
                )
                for segment in segments
            )
        except Exception as exc:
            raise WhisperError(f"transcription failed: {exc}") from exc
        language = getattr(info, "language", None)
        return Transcript(language=None if language is None else str(language), segments=heard)


def _confidence(avg_logprob: object) -> float | None:
    """Turn Whisper's average log-probability into a probability, when it is a number."""
    if not isinstance(avg_logprob, int | float) or not math.isfinite(avg_logprob):
        return None
    return min(1.0, math.exp(avg_logprob))
