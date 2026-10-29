"""Validation of the hyperparameters."""

import pytest

from scene_rag.retrieval.embedding.config import (
    EmbeddingConfig,
    IndexConfig,
    ModelConfig,
    UpstreamConfig,
)


class TestGivenDefaults:
    """The values in force."""

    def test_when_constructed_then_embeddinggemma_cosine_and_any_chunking(self) -> None:
        config = EmbeddingConfig()

        assert config.model.name == "embeddinggemma"
        assert config.model.document_prefix.startswith("title: none")
        assert config.index.distance == "cosine"
        assert config.upstream.chunking_fingerprint is None
        assert config.hash_algorithm == "sha256"

    def test_when_chunking_pinned_then_accepted(self) -> None:
        assert UpstreamConfig(chunking_fingerprint="0123456789ab").chunking_fingerprint


class TestGivenInvalidValues:
    """Values that cannot define a working step."""

    @pytest.mark.parametrize("fingerprint", ["", "0123", "0123456789AB", "ghijklmnopqr"])
    def test_when_pin_malformed_then_value_error(self, fingerprint: str) -> None:
        with pytest.raises(ValueError, match="chunking_fingerprint"):
            UpstreamConfig(chunking_fingerprint=fingerprint)

    def test_when_model_name_empty_then_value_error(self) -> None:
        with pytest.raises(ValueError, match="model name"):
            ModelConfig(name="")

    def test_when_context_below_one_then_value_error(self) -> None:
        with pytest.raises(ValueError, match="num_ctx"):
            ModelConfig(num_ctx=0)

    def test_when_distance_unknown_then_value_error(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            IndexConfig(distance="manhattan")  # type: ignore[arg-type]

    @pytest.mark.parametrize("algorithm", ["md4096", "shake_128"])
    def test_when_hash_algorithm_unusable_then_value_error(self, algorithm: str) -> None:
        with pytest.raises(ValueError, match="hash algorithm"):
            EmbeddingConfig(hash_algorithm=algorithm)
