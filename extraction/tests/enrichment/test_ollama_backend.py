"""The Ollama client and vision model over a fake transport."""

import base64
from typing import Any

import pytest

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import DescriptionConfig
from scene_rag.extraction.enrichment.ollama_backend import OllamaClient, OllamaVisionModel
from scene_rag.extraction.enrichment.ollama_transport import OllamaError

BASE = "http://ollama:11434"
TAGS: dict[str, Any] = {
    "models": [
        {"name": "llama3.2:latest", "model": "llama3.2:latest", "digest": "11" * 32},
        {"name": "qwen2.5vl:3b", "model": "qwen2.5vl:3b", "digest": "ab" * 32},
    ]
}


class FakeTransport:
    """JsonTransport answering from a script keyed by URL path, recording every call."""

    def __init__(self, answers: dict[str, Any], fail: str | None = None) -> None:
        self.answers = answers
        self.fail = fail
        self.calls: list[tuple[str, str, object, float]] = []

    def get(self, url: str, timeout: float) -> dict[str, Any]:
        return self._answer("GET", url, None, timeout)

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return self._answer("POST", url, payload, timeout)

    def _answer(self, method: str, url: str, payload: object, timeout: float) -> dict[str, Any]:
        self.calls.append((method, url, payload, timeout))
        path = url.removeprefix(BASE)
        if self.fail == path:
            raise OllamaError(f"could not reach {url}")
        answer: dict[str, Any] = self.answers[path]
        return answer


class TestGivenReachableServer:
    """A server holding the vision model."""

    @pytest.fixture
    def transport(self) -> FakeTransport:
        return FakeTransport(
            {
                "/api/version": {"version": "0.12.0"},
                "/api/tags": TAGS,
                "/api/chat": {"message": {"role": "assistant", "content": " A talk. "}},
            }
        )

    @pytest.fixture
    def client(self, transport: FakeTransport) -> OllamaClient:
        return OllamaClient(BASE + "/", transport, timeout_seconds=42.0)

    def test_when_version_asked_then_probed_with_short_timeout(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        assert client.version() == "0.12.0"
        assert transport.calls == [("GET", BASE + "/api/version", None, 5.0)]

    def test_when_model_resolved_then_digest_from_tags(self, client: OllamaClient) -> None:
        assert client.resolve_model("qwen2.5vl:3b") == models.ModelRef(
            "qwen2.5vl:3b", "sha256:" + "ab" * 32
        )

    def test_when_model_named_without_tag_then_latest_matches(self, client: OllamaClient) -> None:
        assert client.resolve_model("llama3.2").digest == "sha256:" + "11" * 32

    def test_when_model_missing_then_error_with_pull_hint(self, client: OllamaClient) -> None:
        with pytest.raises(OllamaError, match="ollama pull gemma3:4b"):
            client.resolve_model("gemma3:4b")

    def test_when_described_then_chat_request_carries_images_options_and_no_stream(
        self, client: OllamaClient, transport: FakeTransport
    ) -> None:
        vision = OllamaVisionModel(
            client,
            models.ModelRef("qwen2.5vl:3b", "sha256:" + "ab" * 32),
            DescriptionConfig(temperature=0.1, seed=7, max_tokens=99),
        )

        text = vision.describe([b"img1", b"img2"], "What happens?")

        assert text == " A talk. "
        assert vision.model.name == "qwen2.5vl:3b"
        (method, url, payload, timeout) = transport.calls[-1]
        assert (method, url, timeout) == ("POST", BASE + "/api/chat", 42.0)
        assert payload == {
            "model": "qwen2.5vl:3b",
            "messages": [
                {
                    "role": "user",
                    "content": "What happens?",
                    "images": [
                        base64.b64encode(b"img1").decode(),
                        base64.b64encode(b"img2").decode(),
                    ],
                }
            ],
            "options": {"temperature": 0.1, "seed": 7, "num_predict": 99},
            "stream": False,
        }

    def test_when_chat_answer_malformed_then_error(self, transport: FakeTransport) -> None:
        transport.answers["/api/chat"] = {"done": True}
        client = OllamaClient(BASE, transport, 1.0)

        with pytest.raises(OllamaError, match="no message content"):
            client.chat("qwen2.5vl:3b", "p", [], {})


class TestGivenUnreachableServer:
    """A transport that fails."""

    def test_when_version_asked_then_error_propagates(self) -> None:
        client = OllamaClient(BASE, FakeTransport({}, fail="/api/version"), 1.0)

        with pytest.raises(OllamaError, match="could not reach"):
            client.version()
