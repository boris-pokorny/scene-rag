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

from scene_rag.retrieval.chunking import http_transport
from scene_rag.retrieval.chunking.http_transport import HttpStatusError, UrllibTransport
from scene_rag.retrieval.chunking.protocols import SceneApiError


class Handler(BaseHTTPRequestHandler):
    """Answers a few fixed paths."""

    def do_GET(self) -> None:  # noqa: N802 - the server's naming
        if self.path == "/json":
            body = json.dumps({"status": "ok", "accept": self.headers.get("Accept")}).encode()
            self.send_response(200)
        elif self.path == "/truncated":
            # Promises more bytes than it sends, then hangs up: the client sees IncompleteRead.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"status":')
            self.close_connection = True
            return
        elif self.path == "/garbled":
            # No status line at all: the client sees BadStatusLine.
            self.wfile.write(b"nonsense\r\n\r\n")
            self.close_connection = True
            return
        elif self.path == "/list":
            body = b"[1, 2]"
            self.send_response(200)
        elif self.path == "/text":
            body = b"not json"
            self.send_response(200)
        else:
            body = json.dumps({"error": {"code": "not_found", "message": "nothing"}}).encode()
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
    """A server answering JSON, a list, text and 404."""

    def test_when_json_object_then_decoded_with_accept_header(self, base_url: str) -> None:
        assert UrllibTransport().get(f"{base_url}/json", 5.0) == {
            "status": "ok",
            "accept": "application/json",
        }

    def test_when_error_status_then_http_status_error_with_code(self, base_url: str) -> None:
        with pytest.raises(HttpStatusError, match="HTTP 404") as exc_info:
            UrllibTransport().get(f"{base_url}/missing", 5.0)

        assert exc_info.value.status == 404
        assert "nothing" in str(exc_info.value)

    @pytest.mark.parametrize(
        "path, message", [("/text", "did not answer JSON"), ("/list", "object")]
    )
    def test_when_not_a_json_object_then_scene_api_error(
        self, base_url: str, path: str, message: str
    ) -> None:
        with pytest.raises(SceneApiError, match=message):
            UrllibTransport().get(f"{base_url}{path}", 5.0)

    @pytest.mark.parametrize("path", ["/truncated", "/garbled"])
    def test_when_answer_is_not_http_then_scene_api_error(self, base_url: str, path: str) -> None:
        with pytest.raises(SceneApiError, match="could not reach"):
            UrllibTransport().get(f"{base_url}{path}", 5.0)


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

    def test_when_unreachable_then_scene_api_error(self) -> None:
        with pytest.raises(SceneApiError, match="could not reach"):
            UrllibTransport().get("http://127.0.0.1:1/json", 1.0)
