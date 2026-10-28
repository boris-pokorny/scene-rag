"""Hyperparameters of the enrichment step.

The field defaults *are* the values: changing one is a code change that changes the fingerprint
of the enricher whose section it belongs to. Nothing here comes from the environment;
connection settings live in `settings.py`.
"""

import dataclasses
import hashlib
import math


def _non_negative_finite(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be a non-negative finite number")


@dataclasses.dataclass(frozen=True)
class DescriptionConfig:
    """Tunables of the vision description enricher.

    Attributes:
        model: Vision-language model, as the backend names it; the fingerprint pins its digest.
        prompt: Instruction sent with a scene's keyframes.
        max_keyframes: Most keyframes sent per scene, picked evenly over the scene's frames.
        temperature: Sampling temperature; zero makes equal inputs give equal descriptions.
        seed: Sampling seed, for the same reason.
        max_tokens: Longest description the model may generate, in tokens.
    """

    model: str = "qwen2.5vl:3b"
    prompt: str = (
        "These images are keyframes from a single scene of a video. "
        "Describe what happens in this scene in 2-3 sentences."
    )
    max_keyframes: int = 4
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working enricher.
        """
        if not self.model:
            raise ValueError("description model must not be empty")
        if not self.prompt.strip():
            raise ValueError("description prompt must not be empty")
        if self.max_keyframes < 1:
            raise ValueError("max_keyframes must be at least 1")
        _non_negative_finite("description temperature", self.temperature)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclasses.dataclass(frozen=True)
class TranscriptionConfig:
    """Tunables of the speech transcription enricher.

    Attributes:
        model: Speech model, as the backend names it; the fingerprint pins its digest.
        compute_type: Numeric precision the model runs at; ``int8`` suits CPUs.
        language: Language code to transcribe in; ``None`` detects it per clip.
        beam_size: Beam width of the decoder.
        temperature: Sampling temperature; zero decodes greedily within the beam.
        vad_filter: Whether to skip stretches without speech before decoding.
        condition_on_previous_text: Whether earlier output primes later decoding; off, since a
            clip is one scene and priming carries hallucinations across silences.
    """

    model: str = "small"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = True
    condition_on_previous_text: bool = False

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working enricher.
        """
        if not self.model:
            raise ValueError("transcription model must not be empty")
        if not self.compute_type:
            raise ValueError("compute_type must not be empty")
        if self.language is not None and not self.language:
            raise ValueError("language must be a language code or None")
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")
        _non_negative_finite("transcription temperature", self.temperature)


@dataclasses.dataclass(frozen=True)
class EnrichmentConfig:
    """Every tunable of the enrichment step, and nothing else, one section per enricher.

    Attributes:
        description: Tunables of the vision description enricher.
        transcription: Tunables of the speech transcription enricher.
        hash_algorithm: ``hashlib`` algorithm that versions the records.
    """

    description: DescriptionConfig = DescriptionConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    hash_algorithm: str = "sha256"

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working step.
        """
        if self.hash_algorithm not in hashlib.algorithms_available:
            raise ValueError(f"unknown hash algorithm: {self.hash_algorithm!r}")
        if hashlib.new(self.hash_algorithm).digest_size == 0:  # shake: no fixed-length digest
            raise ValueError(
                f"hash algorithm must have a fixed digest size: {self.hash_algorithm!r}"
            )
