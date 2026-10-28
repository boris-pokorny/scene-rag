"""The faster-whisper backend over a fake model, and the real one when opted in."""

import dataclasses
import io
import math
import wave
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from scene_rag.extraction.enrichment import whisper_backend
from scene_rag.extraction.enrichment.config import TranscriptionConfig
from scene_rag.extraction.enrichment.models import ModelRef
from scene_rag.extraction.enrichment.protocols import ModelError, Transcript, TranscriptSegment
from scene_rag.extraction.enrichment.whisper_backend import FasterWhisperTranscriber, WhisperError


@dataclasses.dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None


@dataclasses.dataclass
class FakeInfo:
    language: str | None


class FakeWhisperModel:
    """Stands in for ``WhisperModel``: canned segments, recorded options, optional failure."""

    def __init__(
        self, segments: list[FakeSegment], language: str | None = "en", fail: bool = False
    ) -> None:
        self.segments = segments
        self.language = language
        self.fail = fail
        self.calls: list[tuple[whisper_backend.Samples, dict[str, object]]] = []

    def transcribe(
        self, audio: whisper_backend.Samples, **options: object
    ) -> tuple[Iterable[Any], Any]:
        self.calls.append((audio, options))
        if self.fail:
            raise ValueError("cannot decode audio")
        return iter(self.segments), FakeInfo(self.language)


MODEL = ModelRef("small", "sha256:" + "cd" * 32)


def pcm_wave(samples: list[int], rate: int = 16000, channels: int = 1, width: int = 2) -> bytes:
    """A WAV file holding ``samples`` (interleaved when several channels)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(width)
        clip.setframerate(rate)
        clip.writeframes(np.array(samples, dtype=f"<i{width}").tobytes())
    return buffer.getvalue()


CLIP = pcm_wave([0, 16384, -16384, 32767])


class TestGivenLoadedModel:
    """A transcriber over a fake model."""

    def test_when_transcribed_then_segments_mapped_and_options_from_config(self) -> None:
        model = FakeWhisperModel(
            [FakeSegment(0.0, 1.2, " Hello ", -0.1), FakeSegment(1.2, 2.0, "world", None)]
        )
        config = TranscriptionConfig(language="cs", beam_size=2, vad_filter=False)

        transcript = FasterWhisperTranscriber(model, MODEL, config).transcribe(CLIP)

        assert transcript.language == "en"
        assert transcript.segments == (
            TranscriptSegment(0.0, 1.2, " Hello ", pytest.approx(math.exp(-0.1))),  # type: ignore[arg-type]
            TranscriptSegment(1.2, 2.0, "world", None),
        )
        ((audio, options),) = model.calls
        assert audio.dtype == np.float32 and audio.tolist() == pytest.approx(
            [0.0, 0.5, -0.5, 32767 / 32768]
        )
        assert options == {
            "language": "cs",
            "beam_size": 2,
            "temperature": 0.0,
            "vad_filter": False,
            "condition_on_previous_text": False,
        }

    def test_when_logprob_positive_or_infinite_then_confidence_capped_or_absent(self) -> None:
        model = FakeWhisperModel(
            [FakeSegment(0.0, 1.0, "a", 0.5), FakeSegment(1.0, 2.0, "b", float("-inf"))]
        )

        transcript = FasterWhisperTranscriber(model, MODEL, TranscriptionConfig()).transcribe(CLIP)

        assert [s.confidence for s in transcript.segments] == [1.0, None]

    def test_when_clip_empty_then_no_segments_and_model_not_called(self) -> None:
        model = FakeWhisperModel([FakeSegment(0.0, 1.0, "ghost", -0.1)])

        transcript = FasterWhisperTranscriber(model, MODEL, TranscriptionConfig()).transcribe(
            pcm_wave([])
        )

        assert transcript == Transcript(language=None, segments=())
        assert model.calls == []

    def test_when_model_fails_then_whisper_error(self) -> None:
        subject = FasterWhisperTranscriber(
            FakeWhisperModel([], fail=True), MODEL, TranscriptionConfig()
        )

        with pytest.raises(WhisperError, match="cannot decode"):
            subject.transcribe(CLIP)
        assert issubclass(WhisperError, ModelError)


class TestGivenClipBytes:
    """Decoding what segmentation files into what the model consumes."""

    def test_when_stereo_then_channels_averaged(self) -> None:
        samples = whisper_backend.decode_pcm_wave(pcm_wave([16384, 0, -16384, -16384], channels=2))

        assert samples.tolist() == pytest.approx([0.25, -0.5])

    @pytest.mark.parametrize(
        "clip, reason",
        [
            (pcm_wave([0, 1], rate=44100), "44100 Hz"),
            (pcm_wave([0, 1], width=1), "8-bit"),
            (b"RIFF not really a wave", "not a WAV"),
            (b"", "not a WAV"),
        ],
    )
    def test_when_not_a_16k_pcm_clip_then_whisper_error(self, clip: bytes, reason: str) -> None:
        with pytest.raises(WhisperError, match=reason):
            whisper_backend.decode_pcm_wave(clip)

    def test_when_model_asked_then_reference_returned(self) -> None:
        assert (
            FasterWhisperTranscriber(FakeWhisperModel([]), MODEL, TranscriptionConfig()).model
            == MODEL
        )
