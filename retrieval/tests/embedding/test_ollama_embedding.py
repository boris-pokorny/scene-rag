"""The Ollama client and embedding model over a fake transport."""

from typing import Any

import pytest

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.config import EmbeddingConfig, ModelConfig
from scene_rag.retrieval.embedding.fingerprint import compute_fingerprint
from scene_rag.retrieval.embedding.ollama_embedding import (
    FINGERPRINTED_MODULES,
    OllamaClient,
    OllamaEmbeddingModel,
)
from scene_rag.retrieval.embedding.protocols import ModelError
from tests.support.embedding_fakes import FakeTransport

BASE = "http://ollama:11434"
DIGEST = "sha256:" + "ab" * 32
TAGS: dict[str, Any] = {
    "models": [
        {
            "name": "nomic-embed-text:latest",
            "model": "nomic-embed-text:latest",
            "digest": "11" * 32,
        },
        {"name": "embeddinggemma:300m", "model": "embeddinggemma:300m", "digest": "ab" * 32},
    ]
}


def embed_answer(payload: dict[str, Any] | None) -> dict[str, Any]:
    assert payload is not None
    return {
        "model": payload["model"],
        "embeddings": [[float(len(text)), 0.5, -0.25] for text in payload["input"]],
        "prompt_eval_count": 7,
    }


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport(
        {
            "GET /api/version": {"version": "0.40.12"},
            "GET /api/tags": TAGS,
            "POST /api/embed": embed_answer,
        },
        base=BASE,
    )


@pytest.fixture
def client(transport: FakeTransport) -> OllamaClient:
    return OllamaClient(BASE + "/", transport, timeout_seconds=42.0)


class TestGivenReachableServer:
    """A server holding the embedding model."""

    def test_when_version_asked_then_probed_with_short_timeout(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        assert client.version() == "0.40.12"
        assert transport.calls == [("GET", BASE + "/api/version", None, 5.0)]

    def test_when_model_resolved_then_digest_from_tags(self, client: OllamaClient) -> None:
        assert client.resolve_model("embeddinggemma:300m") == models.ModelRef(
            "embeddinggemma:300m", DIGEST
        )

    def test_when_model_named_without_tag_then_latest_matches(self, client: OllamaClient) -> None:
        assert client.resolve_model("nomic-embed-text").digest == "sha256:" + "11" * 32

    def test_when_model_missing_then_error_with_pull_hint(self, client: OllamaClient) -> None:
        with pytest.raises(ModelError, match="ollama pull bge-m3"):
            client.resolve_model("bge-m3")

    def test_when_embedded_then_request_carries_prefixed_inputs_truncation_and_context(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        config = EmbeddingConfig(
            model=ModelConfig(name="embeddinggemma:300m", document_prefix="doc: ", num_ctx=512)
        )
        model = OllamaEmbeddingModel(client, models.ModelRef("embeddinggemma:300m", DIGEST), config)

        vectors = model.embed(["one", "three"])

        assert vectors == ((8.0, 0.5, -0.25), (10.0, 0.5, -0.25))
        (method, url, payload, timeout) = transport.calls[-1]
        assert (method, url, timeout) == ("POST", BASE + "/api/embed", 42.0)
        assert payload == {
            "model": "embeddinggemma:300m",
            "input": ["doc: one", "doc: three"],
            "truncate": True,
            "options": {"num_ctx": 512},
        }

    def test_when_constructed_then_fingerprint_pins_config_code_and_digest(
        self, client: OllamaClient
    ) -> None:
        config = EmbeddingConfig()
        model = OllamaEmbeddingModel(client, models.ModelRef("embeddinggemma", DIGEST), config)

        assert model.name == "ollama"
        assert model.model == models.ModelRef("embeddinggemma", DIGEST)
        assert model.fingerprint == compute_fingerprint(config, FINGERPRINTED_MODULES, DIGEST)

    def test_when_dimensions_asked_then_one_probe_then_cached(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        model = OllamaEmbeddingModel(
            client, models.ModelRef("embeddinggemma", DIGEST), EmbeddingConfig()
        )

        assert model.dimensions == 3
        assert model.dimensions == 3
        assert len([c for c in transport.calls if c[0] == "POST"]) == 1

    def test_when_no_texts_then_no_request(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        model = OllamaEmbeddingModel(
            client, models.ModelRef("embeddinggemma", DIGEST), EmbeddingConfig()
        )

        assert model.embed([]) == ()
        assert transport.calls == []

    @pytest.mark.parametrize(
        "answer", [{"done": True}, {"embeddings": [[1.0]]}, {"embeddings": "nope"}]
    )
    def test_when_answer_has_wrong_vector_count_then_error(
        self, client: OllamaClient, transport: FakeTransport, answer: dict[str, Any]
    ) -> None:
        transport.answers["POST /api/embed"] = answer

        with pytest.raises(ModelError, match="vector"):
            client.embed("embeddinggemma", ["a", "b"], truncate=True, num_ctx=1)


class TestGivenUnreachableServer:
    """A transport that fails."""

    def test_when_version_asked_then_model_error(self, transport: FakeTransport) -> None:
        transport.unreachable("GET /api/version")
        client = OllamaClient(BASE, transport, 1.0)

        with pytest.raises(ModelError, match="could not reach"):
            client.version()

    def test_when_status_error_then_model_error(self, transport: FakeTransport) -> None:
        del transport.answers["POST /api/embed"]
        client = OllamaClient(BASE, transport, 1.0)

        with pytest.raises(ModelError, match="HTTP 404"):
            client.embed("embeddinggemma", ["a"], truncate=True, num_ctx=1)
