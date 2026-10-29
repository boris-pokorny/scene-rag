"""Validation of the hyperparameters."""

import dataclasses
import json

import pytest

from scene_rag.retrieval.query_api.config import (
    GeneratorConfig,
    PromptConfig,
    QueryConfig,
    RefusalConfig,
    RerankerConfig,
    RetrieverConfig,
    UpstreamConfig,
)


class TestGivenDefaults:
    """The values in force."""

    def test_when_constructed_then_newest_index_five_sources_no_reranker(self) -> None:
        config = QueryConfig()

        assert config.upstream.embedding_fingerprint is None
        assert config.retriever.top_k == 5
        assert config.retriever.max_top_k >= config.retriever.top_k
        assert config.reranker.name == "none"
        assert 0 < config.refusal.min_score < 1
        assert config.generator.temperature == 0.0
        assert config.generator.name

    def test_when_serialised_then_plain_json(self) -> None:
        assert json.loads(json.dumps(dataclasses.asdict(QueryConfig())))

    def test_when_index_pinned_then_accepted(self) -> None:
        assert UpstreamConfig(embedding_fingerprint="0123456789ab").embedding_fingerprint


class TestGivenInvalidValues:
    """Values that cannot define a working role."""

    @pytest.mark.parametrize("fingerprint", ["", "0123", "0123456789AB", "ghijklmnopqr"])
    def test_when_pin_malformed_then_value_error(self, fingerprint: str) -> None:
        with pytest.raises(ValueError, match="embedding_fingerprint"):
            UpstreamConfig(embedding_fingerprint=fingerprint)

    @pytest.mark.parametrize(
        "field, value",
        [("top_k", 0), ("max_top_k", 4), ("max_question_chars", 0)],
    )
    def test_when_retriever_value_invalid_then_value_error(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            RetrieverConfig(**{field: value})

    def test_when_reranker_unknown_then_value_error(self) -> None:
        with pytest.raises(ValueError, match="reranker"):
            RerankerConfig(name="cross-encoder")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field, value", [("min_score", float("nan")), ("min_score", float("inf")), ("message", "")]
    )
    def test_when_refusal_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            RefusalConfig(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field, value",
        [
            ("user_template", "Question: {question}"),
            ("user_template", "{sources}"),
            ("source_template", "{number}"),
            ("source_template", "{text}"),
        ],
    )
    def test_when_template_lacks_placeholder_then_value_error(self, field: str, value: str) -> None:
        with pytest.raises(ValueError, match=field):
            PromptConfig(**{field: value})

    @pytest.mark.parametrize(
        "field, value",
        [
            ("name", ""),
            ("temperature", -0.1),
            ("temperature", float("nan")),
            ("max_tokens", 0),
            ("num_ctx", 0),
        ],
    )
    def test_when_generator_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            GeneratorConfig(**{field: value})  # type: ignore[arg-type]
