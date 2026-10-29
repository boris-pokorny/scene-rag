"""HTTP layer under the Ollama, Search Index and scene-api clients: the only module speaking HTTP.

Deployment code, deliberately outside the ``query_config`` id: how a request is sent, how
long it waits and how a failure is worded change nothing about what is retrieved or
answered. The transport knows no backend: its errors are translated by each client into its
own.
"""

import http.client
import json
import urllib.error
import urllib.request
from typing import Any, Protocol

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]


class TransportError(RuntimeError):
    """The server could not be reached, or answered something that is not a JSON object."""


class HttpStatusError(TransportError):
    """The server answered with an error status.

    Attributes:
        status: The HTTP status code.
    """

    def __init__(self, status: int, url: str, detail: str) -> None:
        """Name the status and the request that got it."""
        super().__init__(f"HTTP {status} from {url}: {detail}")
        self.status = status


class JsonTransport(Protocol):
    """Exchanges JSON documents over HTTP."""

    def exchange(self, method: str, url: str, payload: Json | None, timeout: float) -> Json:
        """Send ``payload`` (none for a bodiless request) and return the answer.

        Raises:
            HttpStatusError: When the server answers an error status.
            TransportError: When the request fails otherwise or the answer is not a JSON
                object.
        """
        ...


class BytesTransport(Protocol):
    """Fetches raw bodies over HTTP."""

    def fetch(self, url: str, timeout: float) -> tuple[bytes, str]:
        """``GET`` a body and return it with its media type.

        Raises:
            HttpStatusError: When the server answers an error status.
            TransportError: When the request fails otherwise.
        """
        ...


class Transport(JsonTransport, BytesTransport, Protocol):
    """Both: what the scene-api client needs."""


def _detail_of(error: urllib.error.HTTPError) -> str:
    """Return the start of an error body, or nothing when it cannot be read any more."""
    try:
        return error.read().decode(errors="replace")[:500]
    except OSError, http.client.HTTPException:
        return ""


def _send(request: urllib.request.Request, timeout: float) -> tuple[bytes, str]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), str(response.headers.get("Content-Type") or "")
    except urllib.error.HTTPError as exc:
        raise HttpStatusError(exc.code, request.full_url, _detail_of(exc)) from exc
    except (OSError, http.client.HTTPException) as exc:
        # URLError and socket timeouts are OSErrors; a truncated or garbled answer
        # (IncompleteRead, BadStatusLine) is an HTTPException.
        raise TransportError(f"could not reach {request.full_url}: {exc}") from exc


class UrllibTransport:
    """`Transport` over the standard library."""

    def exchange(self, method: str, url: str, payload: Json | None, timeout: float) -> Json:
        """Send one request and decode the JSON answer; an empty body is an empty object."""
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        body, _ = _send(
            urllib.request.Request(url, data=data, headers=headers, method=method), timeout
        )
        if not body:
            return {}
        try:
            document = json.loads(body)
        except ValueError as exc:
            raise TransportError(f"{url} did not answer JSON") from exc
        if not isinstance(document, dict):
            raise TransportError(f"{url} did not answer a JSON object")
        return document

    def fetch(self, url: str, timeout: float) -> tuple[bytes, str]:
        """``GET`` a body with its media type (the ``Content-Type`` without parameters)."""
        body, content_type = _send(urllib.request.Request(url), timeout)
        return body, content_type.split(";", 1)[0].strip()
