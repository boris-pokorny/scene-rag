"""Validation of the hyperparameters."""

import dataclasses

import pytest

from scene_rag.retrieval.chunking.config import ChunkingConfig, PerSceneConfig, UpstreamConfig


class TestGivenDefaults:
    """The shipped values."""

    def test_when_constructed_then_valid_and_frozen(self) -> None:
        config = ChunkingConfig()

        assert config.upstream.enrichment_pins == ()
        assert config.upstream.required_enrichers == ("description", "transcript")
        assert config.per_scene.enrichers == ("description", "transcript")
        assert config.hash_algorithm == "sha256"
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.hash_algorithm = "md5"  # type: ignore[misc]


class TestGivenUpstreamSection:
    """Bounds of the accepted Media outputs."""

    def test_when_pins_valid_then_kept(self) -> None:
        upstream = UpstreamConfig(
            segmentation_fingerprint="3b9e1f6a2c40",
            enrichment_pins=("description:0123456789ab", "transcript:e41a9c27b0d5"),
        )

        assert upstream.segmentation_fingerprint == "3b9e1f6a2c40"

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("segmentation_fingerprint", "xyz", "segmentation_fingerprint"),
            ("enrichment_pins", ("description",), "enrichment_pins"),
            ("enrichment_pins", ("description:xyz",), "twelve hex"),
            ("enrichment_pins", ("ocr:0123456789ab",), "not required"),
            ("enrichment_pins", ("description:0123456789ab", "description:0123456789ac"), "once"),
            ("required_enrichers", (), "at least one"),
            ("required_enrichers", ("description", "description"), "unique"),
            ("required_enrichers", ("Description",), "enricher names"),
        ],
    )
    def test_when_value_invalid_then_value_error(
        self, field: str, value: object, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            UpstreamConfig(**{field: value})  # type: ignore[arg-type]


class TestGivenPerSceneSection:
    """Bounds of the per-scene chunker's tunables."""

    @pytest.mark.parametrize(
        "value, message", [((), "at least one"), (("a", "a"), "unique"), (("9x",), "names")]
    )
    def test_when_enrichers_invalid_then_value_error(self, value: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            PerSceneConfig(enrichers=value)  # type: ignore[arg-type]


class TestGivenStepConfig:
    """Rules across sections."""

    @pytest.mark.parametrize("algorithm", ["no-such-hash", "shake_128"])
    def test_when_hash_unusable_then_value_error(self, algorithm: str) -> None:
        with pytest.raises(ValueError, match="hash algorithm"):
            ChunkingConfig(hash_algorithm=algorithm)

    def test_when_chunker_uses_enricher_not_required_then_value_error(self) -> None:
        with pytest.raises(ValueError, match="ocr"):
            ChunkingConfig(per_scene=PerSceneConfig(enrichers=("description", "ocr")))
