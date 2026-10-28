"""HTTP layer under the Ollama client: the only module speaking HTTP.

Deployment code, deliberately outside the description fingerprint: how a request is sent, how
long a probe waits and how a failure is worded change nothing about what the model answers.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from scene_rag.extraction.enrichment.protocols import ModelError

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]


class OllamaError(ModelError):
    """The Ollama server could not be reached, or refused or failed a request."""


class JsonTransport(Protocol):
    """Exchanges JSON documents over HTTP."""

    def get(self, url: str, timeout: float) -> Json:
        """``GET`` a document.

        Raises:
            OllamaError: When the request fails or the answer is not a JSON object.
        """
        ...

    def post(self, url: str, payload: Json, timeout: float) -> Json:
        """``POST`` a document and return the answer.

        Raises:
            OllamaError: When the request fails or the answer is not a JSON object.
        """
        ...


class UrllibTransport:
    """`JsonTransport` over the standard library."""

    def get(self, url: str, timeout: float) -> Json:
        """``GET`` a JSON document."""
        return self._exchange(urllib.request.Request(url), timeout)

    def post(self, url: str, payload: Json, timeout: float) -> Json:
        """``POST`` a JSON document and decode the JSON answer."""
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._exchange(request, timeout)

    @staticmethod
    def _exchange(request: urllib.request.Request, timeout: float) -> Json:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise OllamaError(f"HTTP {exc.code} from {request.full_url}: {detail}") from exc
        except OSError as exc:  # URLError and socket timeouts are OSErrors
            raise OllamaError(f"could not reach {request.full_url}: {exc}") from exc
        try:
            document = json.loads(body)
        except ValueError as exc:
            raise OllamaError(f"{request.full_url} did not answer JSON") from exc
        if not isinstance(document, dict):
            raise OllamaError(f"{request.full_url} did not answer a JSON object")
        return document
