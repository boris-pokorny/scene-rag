"""Read model over the Search Index the embedding step writes, on Qdrant's REST API.

`QdrantSearchReader` is the role's `SearchIndexReader`. The payload decoder is this role's
copy of the internal contract (``contracts/internal/index-entry.schema.json``);
``tests/contract/`` pins it to the sample. Deployment code, outside the ``query_config``
id: another backend is another module behind the same protocol. The role never writes:
there is no ``PUT`` or ``DELETE`` here.
"""

from collections.abc import Sequence
from typing import Any, Final

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.http_transport import (
    HttpStatusError,
    Json,
    JsonTransport,
    TransportError,
)
from scene_rag.retrieval.query_api.protocols import IndexUnavailableError

_PROBE_TIMEOUT_SECONDS: Final = 5.0
_DISTANCE_NAMES: Final = {"Cosine": "cosine", "Dot": "dot", "Euclid": "euclid"}
ENTITLEMENTS_FIELD: Final = "entitlements"
ASSET_VERSION_FIELD: Final = "asset_version"


def entry_from_payload(payload: Json) -> models.IndexEntryView:
    """Decode a point payload as ``index-entry.schema.json`` describes it.

    Raises:
        ValueError: When the payload's schema version is not the one this role decodes.
        KeyError: When a required field is missing.
    """
    version = payload["schema_version"]
    if version != models.INDEX_ENTRY_SCHEMA_VERSION:
        raise ValueError(
            f"index entry has schema_version {version!r}; this role decodes "
            f"{models.INDEX_ENTRY_SCHEMA_VERSION}"
        )
    return models.IndexEntryView(
        chunk_version=payload["chunk_version"],
        scene_set_version=payload["scene_set_version"],
        asset_version=payload["asset_version"],
        chunking_fingerprint=payload["chunking_fingerprint"],
        scene_indices=tuple(payload["scene_indices"]),
        start_seconds=payload["start_seconds"],
        end_seconds=payload["end_seconds"],
        text=payload["text"],
        sources=tuple(
            models.SourceRef(s["source_id"], s["asset_id"], s["filename"])
            for s in payload["sources"]
        ),
        entitlements=frozenset(payload["entitlements"]),
    )


def search_filter(entitlements: frozenset[str], asset_version: str | None) -> Json:
    """Build the Qdrant filter: an entitlement in common, and the asset version when given."""
    must: list[Json] = [{"key": ENTITLEMENTS_FIELD, "match": {"any": sorted(entitlements)}}]
    if asset_version is not None:
        must.append({"key": ASSET_VERSION_FIELD, "match": {"value": asset_version}})
    return {"must": must}


class QdrantSearchReader:
    """`SearchIndexReader` over one Qdrant server."""

    def __init__(self, base_url: str, transport: JsonTransport, timeout_seconds: float) -> None:
        """Bind to the server at ``base_url``.

        Args:
            base_url: Scheme, host and port of the server, without a path.
            transport: The HTTP layer.
            timeout_seconds: Longest wait for one answer.
        """
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout_seconds

    def _exchange(
        self, method: str, path: str, payload: Json | None = None, timeout: float | None = None
    ) -> Json:
        try:
            return self._transport.exchange(
                method,
                self._base_url + path,
                payload,
                self._timeout if timeout is None else timeout,
            )
        except HttpStatusError:
            raise
        except TransportError as exc:
            raise IndexUnavailableError(str(exc)) from exc

    def _unusable(self, path: str, exc: Exception) -> IndexUnavailableError:
        """Name an answer whose shape is not the one the role reads."""
        return IndexUnavailableError(
            f"{self._base_url}{path} answered something unusable: {type(exc).__name__}: {exc}"
        )

    def version(self) -> str:
        """Return the server version; see `SearchIndexReader.version`."""
        try:
            answer = self._exchange("GET", "/", None, _PROBE_TIMEOUT_SECONDS)
        except HttpStatusError as exc:
            raise IndexUnavailableError(str(exc)) from exc
        return str(answer.get("version", ""))

    def describe(self, collection: str) -> models.CollectionInfo | None:
        """Return the collection's dimensions, distance and size; ``None`` when absent."""
        path = f"/collections/{collection}"
        try:
            answer = self._exchange("GET", path)
        except HttpStatusError as exc:
            if exc.status == 404:
                return None
            raise IndexUnavailableError(str(exc)) from exc
        result = answer.get("result", {})
        try:
            vectors = result["config"]["params"]["vectors"]
            return models.CollectionInfo(
                dimensions=int(vectors["size"]),
                distance=_DISTANCE_NAMES.get(vectors["distance"], str(vectors["distance"]).lower()),
                point_count=int(result.get("points_count") or 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise self._unusable(path, exc) from exc

    def search(
        self,
        collection: str,
        vector: models.Vector,
        *,
        limit: int,
        entitlements: frozenset[str],
        asset_version: str | None,
    ) -> Sequence[models.Hit]:
        """Return the closest visible points, best first; see `SearchIndexReader.search`."""
        if not entitlements:
            return []
        path = f"/collections/{collection}/points/query"
        payload: Json = {
            "query": list(vector),
            "limit": limit,
            "filter": search_filter(entitlements, asset_version),
            "with_payload": True,
            "with_vector": False,
        }
        try:
            answer = self._exchange("POST", path, payload)
        except HttpStatusError as exc:
            raise IndexUnavailableError(str(exc)) from exc
        try:
            points: Any = answer["result"]["points"]
            return [
                models.Hit(entry=entry_from_payload(item["payload"]), score=float(item["score"]))
                for item in points
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise self._unusable(path, exc) from exc
