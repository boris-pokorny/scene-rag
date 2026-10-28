"""Hyperparameters of the segmentation step.

The field defaults *are* the values: changing one is a code change that changes the step's
fingerprint. Nothing here comes from the environment; connection settings live in
`settings.py`.
"""

import dataclasses
import hashlib
import math


def _positive_finite(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number")


@dataclasses.dataclass(frozen=True)
class SegmentationConfig:
    """Every tunable of the segmentation step, and nothing else.

    Attributes:
        scene_threshold: Content detector threshold: the average HSL change between two frames,
            on a 0-255 scale, above which a cut is placed.
        min_scene_seconds: Shortest scene the detector may produce; cuts closer than this to
            the previous one are dropped.
        keyframes_per_scene: Keyframes rendered per scene, spaced evenly over it.
        keyframe_width: Width in pixels keyframes are resized to, keeping the aspect ratio.
        keyframe_frame_margin: Frames kept away from a scene's cuts when picking keyframes,
            so a frame straddling a transition is never chosen.
        keyframe_quality: JPEG quality of the keyframes, 0-100.
        audio_sample_rate: Sample rate in hertz of the per-scene audio clips (16-bit PCM WAV).
        audio_channels: Channel count of the clips; mono suits speech models.
        hash_algorithm: ``hashlib`` algorithm that addresses derivatives and versions scene
            sets.
    """

    scene_threshold: float = 27.0
    min_scene_seconds: float = 0.6
    keyframes_per_scene: int = 2
    keyframe_width: int = 1280
    keyframe_frame_margin: int = 12
    keyframe_quality: int = 95
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    hash_algorithm: str = "sha256"

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working step.
        """
        _positive_finite("scene_threshold", self.scene_threshold)
        if not (math.isfinite(self.min_scene_seconds) and self.min_scene_seconds >= 0):
            raise ValueError("min_scene_seconds must be a non-negative finite number")
        if self.keyframes_per_scene < 1:
            raise ValueError("keyframes_per_scene must be at least 1")
        if self.keyframe_width < 1:
            raise ValueError("keyframe_width must be at least 1")
        if self.keyframe_frame_margin < 0:
            raise ValueError("keyframe_frame_margin must not be negative")
        if not 0 <= self.keyframe_quality <= 100:
            raise ValueError("keyframe_quality must be between 0 and 100")
        if self.audio_sample_rate < 1:
            raise ValueError("audio_sample_rate must be at least 1")
        if self.audio_channels < 1:
            raise ValueError("audio_channels must be at least 1")
        if self.hash_algorithm not in hashlib.algorithms_available:
            raise ValueError(f"unknown hash algorithm: {self.hash_algorithm!r}")
        if hashlib.new(self.hash_algorithm).digest_size == 0:  # shake: no fixed-length digest
            raise ValueError(
                f"hash algorithm must have a fixed digest size: {self.hash_algorithm!r}"
            )
