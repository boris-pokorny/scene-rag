"""The speech transcription enricher over a fake model."""

import pytest

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import TranscriptionConfig
from scene_rag.extraction.enrichment.protocols import (
    DerivativeUnavailableError,
    ModelError,
    Transcript,
)
from scene_rag.extraction.enrichment.transcription import TranscriptionEnricher
from tests.support.enrichment_fakes import (
    SPEECH_MODEL,
    FakeDerivativeReader,
    FakeTranscriber,
    blobs_of,
    clip_bytes,
    heard,
    make_scene_set,
)


def make_enricher(
    scene_set: models.SceneSetView, transcriber: FakeTranscriber | None = None
) -> TranscriptionEnricher:
    return TranscriptionEnricher(
        transcriber=transcriber or FakeTranscriber(),
        derivatives=FakeDerivativeReader(blobs_of(scene_set)),
        config=TranscriptionConfig(),
    )


class TestGivenSceneWithClip:
    """A scene whose audio clip is filed."""

    @pytest.fixture
    def scene_set(self) -> models.SceneSetView:
        return make_scene_set()

    def test_when_enriched_then_segments_offset_into_stream(
        self, scene_set: models.SceneSetView
    ) -> None:
        transcriber = FakeTranscriber(
            Transcript(language="en", segments=(heard("Hello", 0.5, 1.5), heard("there", 1.6, 3.9)))
        )
        scene = scene_set.scenes[1]

        output = make_enricher(scene_set, transcriber).enrich(scene_set, scene)

        assert output.segments == (
            models.SemanticSegment(4.5, 5.5, "Hello", "en", 0.9),
            models.SemanticSegment(5.6, 7.9, "there", "en", 0.9),
        )
        assert scene.audio_clip is not None
        assert output.inputs == (scene.audio_clip.digest,)
        assert transcriber.calls == [clip_bytes(1)]

    def test_when_segments_overrun_scene_then_clamped(self, scene_set: models.SceneSetView) -> None:
        transcriber = FakeTranscriber(
            Transcript(language=None, segments=(heard("late", 3.5, 4.8), heard("later", 4.9, 5.0)))
        )

        output = make_enricher(scene_set, transcriber).enrich(scene_set, scene_set.scenes[0])

        assert [(s.start_seconds, s.end_seconds) for s in output.segments] == [
            (3.5, 4.0),
            (4.0, 4.0),
        ]
        assert output.segments[0].language is None

    def test_when_text_blank_then_segment_dropped(self, scene_set: models.SceneSetView) -> None:
        transcriber = FakeTranscriber(
            Transcript(
                language="en", segments=(heard("  ", 0.0, 1.0), heard(" ok ", 1.0, 2.0, None))
            )
        )

        output = make_enricher(scene_set, transcriber).enrich(scene_set, scene_set.scenes[0])

        assert output.segments == (models.SemanticSegment(1.0, 2.0, "ok", "en", None),)

    def test_when_no_speech_then_no_segments_but_clip_recorded(
        self, scene_set: models.SceneSetView
    ) -> None:
        transcriber = FakeTranscriber(Transcript(language=None, segments=()))

        output = make_enricher(scene_set, transcriber).enrich(scene_set, scene_set.scenes[0])

        assert output.segments == () and len(output.inputs) == 1

    def test_when_identity_asked_then_name_classification_model_and_fingerprint(
        self, scene_set: models.SceneSetView
    ) -> None:
        subject = make_enricher(scene_set)

        assert subject.name == "transcript"
        assert subject.classification == "internal"
        assert subject.model == SPEECH_MODEL
        assert len(subject.fingerprint) == 12

    def test_when_clip_unavailable_then_error_before_model(
        self, scene_set: models.SceneSetView
    ) -> None:
        transcriber = FakeTranscriber()
        subject = TranscriptionEnricher(
            transcriber=transcriber,
            derivatives=FakeDerivativeReader(),
            config=TranscriptionConfig(),
        )

        with pytest.raises(DerivativeUnavailableError):
            subject.enrich(scene_set, scene_set.scenes[0])
        assert transcriber.calls == []

    def test_when_model_fails_then_error_propagates(self, scene_set: models.SceneSetView) -> None:
        with pytest.raises(ModelError):
            make_enricher(scene_set, FakeTranscriber(fail=True)).enrich(
                scene_set, scene_set.scenes[0]
            )


class TestGivenSceneWithoutClip:
    """A scene of an asset without an audio stream."""

    def test_when_enriched_then_empty_output_and_model_not_asked(self) -> None:
        scene_set = make_scene_set(audio=False)
        transcriber = FakeTranscriber()

        output = make_enricher(scene_set, transcriber).enrich(scene_set, scene_set.scenes[0])

        assert output == models.EnricherOutput()
        assert transcriber.calls == []
