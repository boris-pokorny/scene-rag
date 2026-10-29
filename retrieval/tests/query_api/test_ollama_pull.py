"""Pulling the generator model over a fake transport."""

import pytest

from scene_rag.retrieval.query_api.ollama_pull import PULL_TIMEOUT_SECONDS, pull_model
from scene_rag.retrieval.query_api.protocols import ModelError
from tests.support.query_fakes import FakeTransport

BASE = "http://ollama:11434"


class TestGivenServer:
    """A server that pulls, refuses, or is down."""

    def test_when_pulled_then_request_waits_for_the_whole_download(self) -> None:
        transport = FakeTransport({"POST /api/pull": {"status": "success"}}, base=BASE)

        pull_model(BASE + "/", "qwen2.5:3b", transport)

        assert transport.calls == [
            (
                "POST",
                BASE + "/api/pull",
                {"model": "qwen2.5:3b", "stream": False},
                PULL_TIMEOUT_SECONDS,
            )
        ]

    def test_when_status_not_success_then_error(self) -> None:
        transport = FakeTransport({"POST /api/pull": {"status": "pulling manifest"}}, base=BASE)

        with pytest.raises(ModelError, match="pulling manifest"):
            pull_model(BASE, "qwen2.5:3b", transport)

    def test_when_unreachable_then_error(self) -> None:
        transport = FakeTransport(base=BASE)
        transport.unreachable("POST /api/pull")

        with pytest.raises(ModelError, match="could not reach"):
            pull_model(BASE, "qwen2.5:3b", transport)
