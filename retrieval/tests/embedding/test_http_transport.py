"""The standard-library transport against a local HTTP server."""

import email.message
import http.client
import io
import json
import threading
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scene_rag.retrieval.embedding import http_transport
from scene_rag.retrieval.embedding.http_transport import (
    HttpStatusError,
    TransportError,
    UrllibTransport,
)


class Handler(BaseHTTPRequestHandler):
    """Answers a few fixed paths; echoes bodies back."""

    def _answer(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
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
        elif self.path == "/garbled":
            self.wfile.write(b"nonsense\r\n\r\n")
            self.close_connection = True
        elif self.path == "/list":
            self._answer(200, b"[1, 2]")
        elif self.path == "/text":
            self._answer(200, b"not json")
        else:
            self._answer(404, json.dumps({"status": {"error": "nothing here"}}).encode())

    def do_PUT(self) -> None:  # noqa: N802 - the server's naming
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        echo = {
            "method": "PUT",
            "content_type": self.headers.get("Content-Type"),
            "body": json.loads(body),
        }
        self._answer(200, json.dumps(echo).encode())

    def do_DELETE(self) -> None:  # noqa: N802 - the server's naming
        self._answer(200, b'{"result": true}')

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
    """A server answering JSON, an empty body, a list, text and 404."""

    def test_when_got_then_decoded_with_accept_header(self, base_url: str) -> None:
        answer = UrllibTransport().exchange("GET", f"{base_url}/json", None, 5.0)

        assert answer == {"accept": "application/json"}

    def test_when_put_then_body_sent_as_json(self, base_url: str) -> None:
        answer = UrllibTransport().exchange("PUT", f"{base_url}/x", {"a": [1]}, 5.0)

        assert answer == {"method": "PUT", "content_type": "application/json", "body": {"a": [1]}}

    def test_when_deleted_then_answer_decoded(self, base_url: str) -> None:
        assert UrllibTransport().exchange("DELETE", f"{base_url}/x", None, 5.0) == {"result": True}

    def test_when_body_empty_then_empty_object(self, base_url: str) -> None:
        assert UrllibTransport().exchange("GET", f"{base_url}/empty", None, 5.0) == {}

    def test_when_error_status_then_http_status_error_with_code(self, base_url: str) -> None:
        with pytest.raises(HttpStatusError, match="HTTP 404") as exc_info:
            UrllibTransport().exchange("GET", f"{base_url}/missing", None, 5.0)

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

    @pytest.mark.parametrize("path", ["/truncated", "/garbled"])
    def test_when_answer_is_not_http_then_transport_error(self, base_url: str, path: str) -> None:
        with pytest.raises(TransportError, match="could not reach"):
            UrllibTransport().exchange("GET", f"{base_url}{path}", None, 5.0)


class TestGivenUnreadableErrorBody:
    """An error status whose body can no longer be read."""

    def test_when_body_read_fails_then_status_error_without_detail(self) -> None:
        error = _UnreadableError("http://x/", 500, "boom", email.message.Message(), io.BytesIO())

        assert http_transport._detail_of(error) == ""


class _UnreadableError(urllib.error.HTTPError):
    """An error whose connection dropped: reading the body fails."""

    def read(self, amt: int | None = None) -> bytes:
        raise http.client.IncompleteRead(b"partial")


class TestGivenNoServer:
    """A port nobody listens on."""

    def test_when_unreachable_then_transport_error(self) -> None:
        with pytest.raises(TransportError, match="could not reach"):
            UrllibTransport().exchange("GET", "http://127.0.0.1:1/json", None, 1.0)
