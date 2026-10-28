"""The vision description enricher over a fake model."""

import pytest

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import DescriptionConfig
from scene_rag.extraction.enrichment.description import DescriptionEnricher, pick_evenly
from scene_rag.extraction.enrichment.protocols import DerivativeUnavailableError, ModelError
from tests.support.enrichment_fakes import (
    VISION_MODEL,
    FakeDerivativeReader,
    FakeVisionModel,
    blobs_of,
    keyframe_bytes,
    make_scene_set,
)


class TestGivenPickEvenly:
    """Choosing a bounded, even subset of keyframes."""

    def test_when_fewer_than_limit_then_all_kept(self) -> None:
        assert pick_evenly([1, 2, 3], 4) == [1, 2, 3]

    def test_when_more_than_limit_then_first_last_and_even_spacing(self) -> None:
        assert pick_evenly(list(range(9)), 3) == [0, 4, 8]
        assert pick_evenly(list(range(10)), 4) == [0, 3, 6, 9]

    def test_when_limit_is_one_then_middle(self) -> None:
        assert pick_evenly(list(range(7)), 1) == [3]

    def test_when_empty_then_empty(self) -> None:
        assert pick_evenly([], 3) == []


class TestGivenSceneWithKeyframes:
    """A scene of a set, with its keyframes filed."""

    @pytest.fixture
    def scene_set(self) -> models.SceneSetView:
        return make_scene_set(keyframes=6)

    @pytest.fixture
    def enricher(
        self, scene_set: models.SceneSetView
    ) -> tuple[DescriptionEnricher, FakeVisionModel, FakeDerivativeReader]:
        vision = FakeVisionModel()
        derivatives = FakeDerivativeReader(blobs_of(scene_set))
        config = DescriptionConfig(max_keyframes=3)
        return (
            DescriptionEnricher(vision=vision, derivatives=derivatives, config=config),
            vision,
            derivatives,
        )

    def test_when_enriched_then_one_segment_spans_scene_from_even_keyframes(
        self,
        scene_set: models.SceneSetView,
        enricher: tuple[DescriptionEnricher, FakeVisionModel, FakeDerivativeReader],
    ) -> None:
        subject, vision, derivatives = enricher
        scene = scene_set.scenes[1]

        output = subject.enrich(scene_set, scene)

        assert output.segments == (
            models.SemanticSegment(4.0, 8.0, "Seen: jpeg:1:0, jpeg:1:2, jpeg:1:5"),
        )
        assert output.inputs == tuple(scene.keyframes[i].derivative.digest for i in (0, 2, 5))
        assert derivatives.reads == list(output.inputs)
        ((images, prompt),) = vision.calls
        assert images == (keyframe_bytes(1, 0), keyframe_bytes(1, 2), keyframe_bytes(1, 5))
        assert prompt == DescriptionConfig().prompt

    def test_when_identity_asked_then_name_classification_model_and_fingerprint(
        self, enricher: tuple[DescriptionEnricher, FakeVisionModel, FakeDerivativeReader]
    ) -> None:
        subject, _, _ = enricher

        assert subject.name == "description"
        assert subject.classification == "internal"
        assert subject.model == VISION_MODEL
        assert len(subject.fingerprint) == 12

    def test_when_model_answers_blank_then_no_segment_but_inputs_recorded(
        self, scene_set: models.SceneSetView
    ) -> None:
        subject = DescriptionEnricher(
            vision=FakeVisionModel("  \n"),
            derivatives=FakeDerivativeReader(blobs_of(scene_set)),
            config=DescriptionConfig(max_keyframes=2),
        )

        output = subject.enrich(scene_set, scene_set.scenes[0])

        assert output.segments == ()
        assert len(output.inputs) == 2

    def test_when_model_answer_padded_then_stripped(self, scene_set: models.SceneSetView) -> None:
        subject = DescriptionEnricher(
            vision=FakeVisionModel(" A talk.\n"),
            derivatives=FakeDerivativeReader(blobs_of(scene_set)),
            config=DescriptionConfig(),
        )

        assert subject.enrich(scene_set, scene_set.scenes[0]).segments[0].text == "A talk."

    def test_when_keyframe_unavailable_then_error_before_model(
        self, scene_set: models.SceneSetView
    ) -> None:
        vision = FakeVisionModel()
        subject = DescriptionEnricher(
            vision=vision, derivatives=FakeDerivativeReader(), config=DescriptionConfig()
        )

        with pytest.raises(DerivativeUnavailableError):
            subject.enrich(scene_set, scene_set.scenes[0])
        assert vision.calls == []

    def test_when_model_fails_then_error_propagates(self, scene_set: models.SceneSetView) -> None:
        subject = DescriptionEnricher(
            vision=FakeVisionModel(fail=True),
            derivatives=FakeDerivativeReader(blobs_of(scene_set)),
            config=DescriptionConfig(),
        )

        with pytest.raises(ModelError):
            subject.enrich(scene_set, scene_set.scenes[0])


class TestGivenSceneWithoutKeyframes:
    """A scene nothing could be rendered for."""

    def test_when_enriched_then_empty_output_and_model_not_asked(self) -> None:
        scene_set = make_scene_set(keyframes=0)
        vision = FakeVisionModel()
        subject = DescriptionEnricher(
            vision=vision, derivatives=FakeDerivativeReader(), config=DescriptionConfig()
        )

        assert subject.enrich(scene_set, scene_set.scenes[0]) == models.EnricherOutput()
        assert vision.calls == []
