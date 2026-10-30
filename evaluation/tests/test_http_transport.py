"""The standard-library transport against a local HTTP server."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scene_rag.evaluation.http_transport import (
    HttpStatusError,
    TransportError,
    UrllibTransport,
)


class Handler(BaseHTTPRequestHandler):
    """Answers a few fixed paths; echoes bodies and headers back."""

    def _answer(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - the server's naming
        if self.path == "/empty":
            self._answer(200, b"")
        elif self.path == "/list":
            self._answer(200, b"[1, 2]")
        elif self.path == "/text":
            self._answer(200, b"not json")
        else:
            self._answer(404, json.dumps({"error": "nothing here"}).encode())

    def do_POST(self) -> None:  # noqa: N802 - the server's naming
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        echo = {
            "content_type": self.headers.get("Content-Type"),
            "accept": self.headers.get("Accept"),
            "entitlements": self.headers.get("X-Scene-Rag-Entitlements"),
            "body": json.loads(body),
        }
        self._answer(200, json.dumps(echo).encode())

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the server's naming
        return


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


class TestGivenServer:
    """A server echoing posts, and answering an empty body, a list, text and 404."""

    def test_when_posted_then_body_and_headers_sent(self, base_url: str) -> None:
        answer = UrllibTransport().exchange(
            "POST", f"{base_url}/x", {"a": 1}, {"X-Scene-Rag-Entitlements": "public"}, 5.0
        )

        assert answer == {
            "content_type": "application/json",
            "accept": "application/json",
            "entitlements": "public",
            "body": {"a": 1},
        }

    def test_when_body_empty_then_empty_object(self, base_url: str) -> None:
        assert UrllibTransport().exchange("GET", f"{base_url}/empty", None, {}, 5.0) == {}

    @pytest.mark.parametrize(("path", "message"), [("/list", "JSON object"), ("/text", "JSON")])
    def test_when_not_json_object_then_transport_error(
        self, base_url: str, path: str, message: str
    ) -> None:
        with pytest.raises(TransportError, match=message):
            UrllibTransport().exchange("GET", f"{base_url}{path}", None, {}, 5.0)

    def test_when_error_status_then_status_and_detail(self, base_url: str) -> None:
        with pytest.raises(HttpStatusError, match="nothing here") as info:
            UrllibTransport().exchange("GET", f"{base_url}/missing", None, {}, 5.0)
        assert info.value.status == 404

    def test_when_unreachable_then_transport_error(self) -> None:
        with pytest.raises(TransportError, match="could not reach"):
            UrllibTransport().exchange("GET", "http://127.0.0.1:1/x", None, {}, 1.0)
