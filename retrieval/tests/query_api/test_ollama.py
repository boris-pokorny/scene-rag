"""The Ollama client, query embedder and answer generator over a fake transport."""

from typing import Any

import pytest

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.config import GeneratorConfig
from scene_rag.retrieval.query_api.ollama import (
    OllamaAnswerGenerator,
    OllamaClient,
    OllamaQueryEmbedder,
)
from scene_rag.retrieval.query_api.protocols import ModelError
from tests.support.query_fakes import (
    EMBEDDING_MODEL,
    GENERATOR_MODEL,
    FakeTransport,
    make_index_view,
)

BASE = "http://ollama:11434"
TAGS: dict[str, Any] = {
    "models": [
        {"name": "qwen2.5:3b", "model": "qwen2.5:3b", "digest": "cd" * 32},
        {"name": "embeddinggemma:latest", "model": "embeddinggemma:latest", "digest": "ab" * 32},
    ]
}


def embed_answer(payload: dict[str, Any] | None) -> dict[str, Any]:
    assert payload is not None
    return {"embeddings": [[float(len(text)), 0.5, -0.25] for text in payload["input"]]}


def chat_answer(payload: dict[str, Any] | None) -> dict[str, Any]:
    assert payload is not None
    return {
        "message": {"role": "assistant", "content": f"echo: {payload['messages'][-1]['content']}"},
        "prompt_eval_count": 40,
        "eval_count": 9,
    }


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport(
        {
            "GET /api/version": {"version": "0.40.12"},
            "GET /api/tags": TAGS,
            "POST /api/embed": embed_answer,
            "POST /api/chat": chat_answer,
        },
        base=BASE,
    )


@pytest.fixture
def client(transport: FakeTransport) -> OllamaClient:
    return OllamaClient(
        BASE + "/", transport, embed_timeout_seconds=42.0, chat_timeout_seconds=99.0
    )


class TestGivenReachableServer:
    """A server holding both models."""

    def test_when_version_asked_then_probed_with_short_timeout(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        assert client.version() == "0.40.12"
        assert client.base_url == BASE
        assert transport.calls == [("GET", BASE + "/api/version", None, 5.0)]

    def test_when_model_resolved_then_digest_from_tags(self, client: OllamaClient) -> None:
        assert client.resolve_model("qwen2.5:3b") == GENERATOR_MODEL
        assert client.resolve_model("embeddinggemma") == EMBEDDING_MODEL

    def test_when_model_missing_then_error_with_pull_hint(self, client: OllamaClient) -> None:
        with pytest.raises(ModelError, match="ollama pull llama3"):
            client.resolve_model("llama3")

    def test_when_question_embedded_then_index_prefix_truncation_and_context(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        index = make_index_view(dimensions=3)
        embedder = OllamaQueryEmbedder(client, index)

        vector = embedder.embed("when?")

        assert vector == (34.0, 0.5, -0.25)
        assert (embedder.model, embedder.dimensions) == (EMBEDDING_MODEL, 3)
        assert transport.calls[-1] == (
            "POST",
            BASE + "/api/embed",
            {
                "model": "embeddinggemma",
                "input": ["task: search result | query: when?"],
                "truncate": True,
                "options": {"num_ctx": 2048},
            },
            42.0,
        )

    def test_when_vector_length_differs_from_index_then_error(self, client: OllamaClient) -> None:
        with pytest.raises(ModelError, match="3 dimensions"):
            OllamaQueryEmbedder(client, make_index_view(dimensions=768)).embed("q")

    @pytest.mark.parametrize("answer", [{"done": True}, {"embeddings": []}, {"embeddings": "x"}])
    def test_when_embed_answer_malformed_then_error(
        self, client: OllamaClient, transport: FakeTransport, answer: dict[str, Any]
    ) -> None:
        transport.answers["POST /api/embed"] = answer

        with pytest.raises(ModelError, match="vector"):
            client.embed("embeddinggemma", ["a"], truncate=True, num_ctx=1)

    def test_when_generated_then_chat_request_carries_messages_and_sampling(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        config = GeneratorConfig(temperature=0.2, seed=7, max_tokens=64, num_ctx=4096)
        generator = OllamaAnswerGenerator(client, GENERATOR_MODEL, config)
        messages = [models.Message("system", "be brief"), models.Message("user", "hi")]

        generation = generator.generate(messages)

        assert generation == models.Generation("echo: hi", models.Usage(40, 9))
        assert generator.model == GENERATOR_MODEL
        assert transport.calls[-1] == (
            "POST",
            BASE + "/api/chat",
            {
                "model": "qwen2.5:3b",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ],
                "options": {"temperature": 0.2, "seed": 7, "num_predict": 64, "num_ctx": 4096},
                "stream": False,
            },
            99.0,
        )

    def test_when_usage_missing_then_none(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        transport.answers["POST /api/chat"] = {"message": {"content": "ok"}, "eval_count": True}

        generation = client.chat("qwen2.5:3b", [models.Message("user", "hi")], {})

        assert generation.usage == models.Usage(None, None)

    @pytest.mark.parametrize("answer", [{"done": True}, {"message": "x"}, {"message": {"c": 1}}])
    def test_when_chat_answer_malformed_then_error(
        self, client: OllamaClient, transport: FakeTransport, answer: dict[str, Any]
    ) -> None:
        transport.answers["POST /api/chat"] = answer

        with pytest.raises(ModelError, match="no message content"):
            client.chat("qwen2.5:3b", [models.Message("user", "hi")], {})


class TestGivenUnreachableServer:
    """A transport that fails."""

    def test_when_version_asked_then_model_error(self, transport: FakeTransport) -> None:
        transport.unreachable("GET /api/version")
        client = OllamaClient(BASE, transport, embed_timeout_seconds=1.0, chat_timeout_seconds=1.0)

        with pytest.raises(ModelError, match="could not reach"):
            client.version()

    def test_when_status_error_then_model_error(self, transport: FakeTransport) -> None:
        del transport.answers["POST /api/chat"]
        client = OllamaClient(BASE, transport, embed_timeout_seconds=1.0, chat_timeout_seconds=1.0)

        with pytest.raises(ModelError, match="HTTP 404"):
            client.chat("qwen2.5:3b", [], {})
