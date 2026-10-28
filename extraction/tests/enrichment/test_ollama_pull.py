"""Pulling a model through the fake transport: the request sent and the answer accepted."""

from typing import Any

import pytest

from scene_rag.extraction.enrichment import ollama_pull
from scene_rag.extraction.enrichment.ollama_transport import OllamaError


class RecordingTransport:
    """JsonTransport answering every POST with one scripted document."""

    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.posts: list[tuple[str, dict[str, Any], float]] = []

    def get(self, url: str, timeout: float) -> dict[str, Any]:
        raise AssertionError("a pull never reads")

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.posts.append((url, payload, timeout))
        return self.answer


class TestGivenServer:
    """A server answering the pull endpoint."""

    def test_when_pulled_then_one_blocking_request_under_the_long_timeout(self) -> None:
        transport = RecordingTransport({"status": "success"})

        ollama_pull.pull_model("http://ollama:11434/", "qwen2.5vl:3b", transport)

        assert transport.posts == [
            (
                "http://ollama:11434/api/pull",
                {"model": "qwen2.5vl:3b", "stream": False},
                ollama_pull.PULL_TIMEOUT_SECONDS,
            )
        ]

    @pytest.mark.parametrize("answer", [{"status": "error"}, {}])
    def test_when_answer_is_not_success_then_error_names_model(
        self, answer: dict[str, Any]
    ) -> None:
        with pytest.raises(OllamaError, match="qwen2.5vl:3b"):
            ollama_pull.pull_model(
                "http://ollama:11434", "qwen2.5vl:3b", RecordingTransport(answer)
            )
