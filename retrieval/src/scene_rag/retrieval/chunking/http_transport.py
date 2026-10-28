"""HTTP layer under the scene-api client: the only module speaking HTTP.

Deployment code, deliberately outside the fingerprint: how a request is sent, how long it
waits and how a failure is worded change nothing about what a chunk contains.
"""

import http.client
import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from scene_rag.retrieval.chunking.protocols import SceneApiError

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]


class HttpStatusError(SceneApiError):
    """The server answered with an error status.

    Attributes:
        status: The HTTP status code.
    """

    def __init__(self, status: int, url: str, detail: str) -> None:
        """Name the status and the request that got it."""
        super().__init__(f"HTTP {status} from {url}: {detail}")
        self.status = status


class JsonTransport(Protocol):
    """Fetches JSON documents over HTTP."""

    def get(self, url: str, timeout: float) -> Json:
        """``GET`` a document.

        Raises:
            HttpStatusError: When the server answers an error status.
            SceneApiError: When the request fails otherwise or the answer is not a JSON object.
        """
        ...


def _detail_of(error: urllib.error.HTTPError) -> str:
    """Return the start of an error body, or nothing when it cannot be read any more."""
    try:
        return error.read().decode(errors="replace")[:500]
    except OSError, http.client.HTTPException:
        return ""


class UrllibTransport:
    """`JsonTransport` over the standard library."""

    def get(self, url: str, timeout: float) -> Json:
        """``GET`` a JSON document."""
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise HttpStatusError(exc.code, url, _detail_of(exc)) from exc
        except (OSError, http.client.HTTPException) as exc:
            # URLError and socket timeouts are OSErrors; a truncated or garbled answer
            # (IncompleteRead, BadStatusLine) is an HTTPException.
            raise SceneApiError(f"could not reach {url}: {exc}") from exc
        try:
            document = json.loads(body)
        except ValueError as exc:
            raise SceneApiError(f"{url} did not answer JSON") from exc
        if not isinstance(document, dict):
            raise SceneApiError(f"{url} did not answer a JSON object")
        return document
