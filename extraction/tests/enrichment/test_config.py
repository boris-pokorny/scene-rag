"""Validation of the hyperparameters."""

import dataclasses

import pytest

from scene_rag.extraction.enrichment.config import (
    DescriptionConfig,
    EnrichmentConfig,
    TranscriptionConfig,
)


class TestGivenDefaults:
    """The shipped values."""

    def test_when_constructed_then_valid_and_frozen(self) -> None:
        config = EnrichmentConfig()

        assert config.description.model == "qwen2.5vl:3b"
        assert config.transcription.model == "small"
        assert config.hash_algorithm == "sha256"
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.hash_algorithm = "md5"  # type: ignore[misc]


class TestGivenDescriptionSection:
    """Bounds of the vision enricher's tunables."""

    @pytest.mark.parametrize(
        "field, value",
        [
            ("model", ""),
            ("prompt", "   "),
            ("max_keyframes", 0),
            ("temperature", -0.1),
            ("temperature", float("nan")),
            ("max_tokens", 0),
        ],
    )
    def test_when_value_out_of_range_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            DescriptionConfig(**{field: value})  # type: ignore[arg-type]


class TestGivenTranscriptionSection:
    """Bounds of the speech enricher's tunables."""

    @pytest.mark.parametrize(
        "field, value",
        [
            ("model", ""),
            ("compute_type", ""),
            ("language", ""),
            ("beam_size", 0),
            ("temperature", -1.0),
        ],
    )
    def test_when_value_out_of_range_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            TranscriptionConfig(**{field: value})  # type: ignore[arg-type]

    def test_when_language_given_then_kept(self) -> None:
        assert TranscriptionConfig(language="cs").language == "cs"


class TestGivenHashAlgorithm:
    """The algorithm versioning the records."""

    @pytest.mark.parametrize("algorithm", ["no-such-hash", "shake_128"])
    def test_when_unusable_then_value_error(self, algorithm: str) -> None:
        with pytest.raises(ValueError, match="hash algorithm"):
            EnrichmentConfig(hash_algorithm=algorithm)
