"""Pulling the model over a fake transport."""

import pytest

from scene_rag.retrieval.embedding import ollama_pull
from scene_rag.retrieval.embedding.protocols import ModelError
from tests.support.embedding_fakes import FakeTransport

BASE = "http://ollama:11434"


class TestGivenServer:
    """A server that pulls, refuses, or is down."""

    def test_when_pulled_then_post_without_stream_and_long_timeout(self) -> None:
        transport = FakeTransport({"POST /api/pull": {"status": "success"}}, base=BASE)

        ollama_pull.pull_model(BASE + "/", "embeddinggemma", transport)

        assert transport.calls == [
            ("POST", BASE + "/api/pull", {"model": "embeddinggemma", "stream": False}, 3600.0)
        ]

    def test_when_status_not_success_then_error(self) -> None:
        transport = FakeTransport({"POST /api/pull": {"status": "pulling manifest"}}, base=BASE)

        with pytest.raises(ModelError, match="pulling manifest"):
            ollama_pull.pull_model(BASE, "embeddinggemma", transport)

    def test_when_unreachable_then_model_error(self) -> None:
        transport = FakeTransport(base=BASE)
        transport.unreachable("POST /api/pull")

        with pytest.raises(ModelError, match="could not reach"):
            ollama_pull.pull_model(BASE, "embeddinggemma", transport)
