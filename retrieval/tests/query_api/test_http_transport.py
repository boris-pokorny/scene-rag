"""The standard-library transport against a local HTTP server."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scene_rag.retrieval.query_api.http_transport import (
    HttpStatusError,
    TransportError,
    UrllibTransport,
)


class Handler(BaseHTTPRequestHandler):
    """Answers a few fixed paths; echoes bodies back."""

    def _answer(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - the server's naming
        if self.path == "/json":
            self._answer(200, json.dumps({"accept": self.headers.get("Accept")}).encode())
        elif self.path == "/empty":
            self._answer(200, b"")
        elif self.path == "/truncated":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"status":')
            self.close_connection = True
        elif self.path == "/list":
            self._answer(200, b"[1, 2]")
        elif self.path == "/text":
            self._answer(200, b"not json")
        elif self.path == "/image":
            self._answer(200, b"\xff\xd8jpeg", "image/jpeg; charset=binary")
        else:
            self._answer(404, json.dumps({"error": "nothing here"}).encode())

    def do_POST(self) -> None:  # noqa: N802 - the server's naming
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        echo = {"content_type": self.headers.get("Content-Type"), "body": json.loads(body)}
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
    """A server answering JSON, bytes, an empty body, a list, text and 404."""

    def test_when_got_then_decoded_with_accept_header(self, base_url: str) -> None:
        assert UrllibTransport().exchange("GET", f"{base_url}/json", None, 5.0) == {
            "accept": "application/json"
        }

    def test_when_posted_then_body_sent_as_json(self, base_url: str) -> None:
        answer = UrllibTransport().exchange("POST", f"{base_url}/x", {"a": [1]}, 5.0)

        assert answer == {"content_type": "application/json", "body": {"a": [1]}}

    def test_when_body_empty_then_empty_object(self, base_url: str) -> None:
        assert UrllibTransport().exchange("GET", f"{base_url}/empty", None, 5.0) == {}

    def test_when_fetched_then_bytes_with_bare_media_type(self, base_url: str) -> None:
        assert UrllibTransport().fetch(f"{base_url}/image", 5.0) == (b"\xff\xd8jpeg", "image/jpeg")

    @pytest.mark.parametrize("method", ["exchange", "fetch"])
    def test_when_error_status_then_http_status_error_with_code(
        self, base_url: str, method: str
    ) -> None:
        transport = UrllibTransport()
        with pytest.raises(HttpStatusError, match="HTTP 404") as exc_info:
            if method == "fetch":
                transport.fetch(f"{base_url}/missing", 5.0)
            else:
                transport.exchange("GET", f"{base_url}/missing", None, 5.0)

        assert exc_info.value.status == 404
        assert "nothing here" in str(exc_info.value)

    @pytest.mark.parametrize(
        "path, message", [("/text", "did not answer JSON"), ("/list", "object")]
    )
    def test_when_not_a_json_object_then_transport_error(
        self, base_url: str, path: str, message: str
    ) -> None:
        with pytest.raises(TransportError, match=message):
            UrllibTransport().exchange("GET", f"{base_url}{path}", None, 5.0)

    def test_when_answer_truncated_then_transport_error(self, base_url: str) -> None:
        with pytest.raises(TransportError, match="could not reach"):
            UrllibTransport().exchange("GET", f"{base_url}/truncated", None, 5.0)


class TestGivenNoServer:
    """A port nobody listens on."""

    def test_when_unreachable_then_transport_error(self) -> None:
        with pytest.raises(TransportError, match="could not reach"):
            UrllibTransport().fetch("http://127.0.0.1:1/json", 1.0)
