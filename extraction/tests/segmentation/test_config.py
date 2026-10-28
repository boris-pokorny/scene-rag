from collections.abc import Callable

import pytest

from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.fingerprint import compute_fingerprint


class TestGivenConfig:
    """The step's hyperparameters and their fingerprint."""

    def test_when_defaults_then_poc_values_kept(self) -> None:
        config = SegmentationConfig()
        assert config.scene_threshold == 27.0
        assert config.keyframes_per_scene == 2
        assert config.audio_sample_rate == 16000
        assert config.hash_algorithm == "sha256"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: SegmentationConfig(scene_threshold=0.0),
            lambda: SegmentationConfig(scene_threshold=float("nan")),
            lambda: SegmentationConfig(min_scene_seconds=-1.0),
            lambda: SegmentationConfig(min_scene_seconds=float("inf")),
            lambda: SegmentationConfig(keyframes_per_scene=0),
            lambda: SegmentationConfig(keyframe_width=0),
            lambda: SegmentationConfig(keyframe_frame_margin=-1),
            lambda: SegmentationConfig(keyframe_quality=101),
            lambda: SegmentationConfig(audio_sample_rate=0),
            lambda: SegmentationConfig(audio_channels=0),
            lambda: SegmentationConfig(hash_algorithm="no-such-hash"),
            lambda: SegmentationConfig(hash_algorithm="shake_128"),
        ],
    )
    def test_when_value_invalid_then_rejected(
        self, build: Callable[[], SegmentationConfig]
    ) -> None:
        with pytest.raises(ValueError):
            build()

    def test_when_fingerprinted_twice_then_equal(self) -> None:
        assert compute_fingerprint(SegmentationConfig()) == compute_fingerprint(
            SegmentationConfig()
        )
        assert len(compute_fingerprint(SegmentationConfig())) == 12

    def test_when_hyperparameter_changes_then_fingerprint_differs(self) -> None:
        assert compute_fingerprint(SegmentationConfig()) != compute_fingerprint(
            SegmentationConfig(scene_threshold=30.0)
        )
