"""The standard-library transport over a patched ``urlopen``."""

import json

import pytest

from scene_rag.extraction.enrichment.ollama_transport import OllamaError, UrllibTransport

BASE = "http://ollama:11434"


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TestGivenUrllibTransport:
    """The standard-library transport over a patched ``urlopen``."""

    def test_when_posted_then_json_request_and_decoded_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.request

        seen: list[tuple[urllib.request.Request, float]] = []

        def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
            seen.append((request, timeout))
            return FakeResponse(b'{"ok": 1}')

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)

        answer = UrllibTransport().post(BASE + "/api/chat", {"a": 1}, 3.0)

        assert answer == {"ok": 1}
        ((request, timeout),) = seen
        assert request.get_method() == "POST" and timeout == 3.0
        assert json.loads(request.data or b"") == {"a": 1}  # type: ignore[arg-type]
        assert request.get_header("Content-type") == "application/json"

    def test_when_get_then_method_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request, timeout: FakeResponse(b'{"version": "1"}')
        )

        assert UrllibTransport().get(BASE + "/api/version", 1.0) == {"version": "1"}

    @pytest.mark.parametrize("body", [b"not json", b"[1, 2]"])
    def test_when_answer_not_object_then_error(
        self, monkeypatch: pytest.MonkeyPatch, body: bytes
    ) -> None:
        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse(body))

        with pytest.raises(OllamaError, match="did not answer"):
            UrllibTransport().get(BASE + "/api/version", 1.0)

    def test_when_http_error_then_status_and_body_in_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import email.message
        import io
        import urllib.error
        import urllib.request

        def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
            raise urllib.error.HTTPError(
                request.full_url,
                404,
                "Not Found",
                email.message.Message(),
                io.BytesIO(b'{"error":"model not found"}'),
            )

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)

        with pytest.raises(OllamaError, match="HTTP 404.*model not found"):
            UrllibTransport().get(BASE + "/api/tags", 1.0)

    def test_when_connection_refused_then_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.error
        import urllib.request

        def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)

        with pytest.raises(OllamaError, match="could not reach"):
            UrllibTransport().get(BASE + "/api/tags", 1.0)
