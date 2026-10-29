"""Read model over the scene-api of the extraction service.

`SceneApiClient` is the role's `SceneReader`: it fetches one scene record, passed through
as the published document (``scene-record.schema.json`` of extraction) after checking its
schema version, and the bytes of one derivative. The HTTP layer lives in `http_transport`.
"""

import http
from typing import Final
from urllib.parse import quote

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.http_transport import HttpStatusError, Transport, TransportError
from scene_rag.retrieval.query_api.protocols import Json, SceneApiError

_PROBE_TIMEOUT_SECONDS: Final = 5.0


class SceneApiClient:
    """`SceneReader` over the scene-api."""

    def __init__(self, base_url: str, transport: Transport, timeout_seconds: float) -> None:
        """Bind to the API at ``base_url``.

        Args:
            base_url: Scheme, host, port (and path prefix, if any) of the API.
            transport: The HTTP layer.
            timeout_seconds: Longest wait for one answer.
        """
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout_seconds

    def health(self) -> None:
        """Check that the API answers; see `SceneReader.health`."""
        url = f"{self._base_url}/health"
        try:
            answer = self._transport.exchange("GET", url, None, _PROBE_TIMEOUT_SECONDS)
        except TransportError as exc:
            raise SceneApiError(str(exc)) from exc
        if answer.get("status") != "ok":
            raise SceneApiError(f"{url} reports {answer!r}")

    def get_scene_record(self, scene_set_version: str, index: int) -> Json | None:
        """Return the published scene record; see `SceneReader.get_scene_record`."""
        url = f"{self._base_url}/scene-sets/{quote(scene_set_version, safe=':')}/scenes/{index}"
        try:
            document = self._transport.exchange("GET", url, None, self._timeout)
        except HttpStatusError as exc:
            if exc.status == http.HTTPStatus.NOT_FOUND:
                return None
            raise SceneApiError(str(exc)) from exc
        except TransportError as exc:
            raise SceneApiError(str(exc)) from exc
        version = document.get("schema_version")
        if version != models.SCENE_RECORD_SCHEMA_VERSION:
            raise SceneApiError(
                f"scene record has schema_version {version!r}; this role serves "
                f"{models.SCENE_RECORD_SCHEMA_VERSION}"
            )
        return document

    def get_derivative(self, digest: str) -> models.DerivativeContent | None:
        """Return the bytes of a derivative; see `SceneReader.get_derivative`."""
        url = f"{self._base_url}/derivatives/{quote(digest, safe=':')}"
        try:
            data, media_type = self._transport.fetch(url, self._timeout)
        except HttpStatusError as exc:
            if exc.status == http.HTTPStatus.NOT_FOUND:
                return None
            raise SceneApiError(str(exc)) from exc
        except TransportError as exc:
            raise SceneApiError(str(exc)) from exc
        return models.DerivativeContent(media_type=media_type, data=data)
