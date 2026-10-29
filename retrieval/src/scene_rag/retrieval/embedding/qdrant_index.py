"""Search Index backend over Qdrant's REST API.

`QdrantSearchIndex` implements `SearchIndexReader` and `SearchIndexWriter`. A collection
holds one embedding fingerprint; a point's id is the UUID of its chunk version and its
payload the index entry (``contracts/internal/index-entry.schema.json``). Deployment code,
outside the fingerprint: another backend is another module behind the same protocols.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.http_transport import (
    HttpStatusError,
    Json,
    JsonTransport,
    TransportError,
)
from scene_rag.retrieval.embedding.protocols import IndexMismatchError, SearchIndexError

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS: Final = 5.0
_DISTANCES: Final = {"cosine": "Cosine", "dot": "Dot", "euclid": "Euclid"}
_DISTANCE_NAMES: Final = {qdrant: ours for ours, qdrant in _DISTANCES.items()}
# Payload fields query-api filters on: entitlements, the retired-asset block list, purge.
PAYLOAD_INDEXES: Final = ("entitlements", "asset_version", "chunk_set_version")


def _encode_datetime(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def entry_to_payload(entry: models.IndexEntry) -> Json:
    """Encode an entry as a point payload (``contracts/internal/index-entry.schema.json``)."""
    return {
        "version": entry.version,
        "schema_version": entry.schema_version,
        "chunk_version": entry.chunk_version,
        "chunk_set_version": entry.chunk_set_version,
        "scene_set_version": entry.scene_set_version,
        "asset_version": entry.asset_version,
        "chunking_fingerprint": entry.chunking_fingerprint,
        "segmentation_fingerprint": entry.segmentation_fingerprint,
        "chunk_index": entry.chunk_index,
        "scene_indices": list(entry.scene_indices),
        "start_seconds": entry.start_seconds,
        "end_seconds": entry.end_seconds,
        "text": entry.text,
        "sources": [
            {"source_id": s.source_id, "asset_id": s.asset_id, "filename": s.filename}
            for s in entry.sources
        ],
        "entitlements": sorted(entry.entitlements),
        "classifications": list(entry.classifications),
        "created_at": _encode_datetime(entry.created_at),
        "lineage": {
            "step": entry.lineage.step,
            "fingerprint": entry.lineage.fingerprint,
            "inputs": list(entry.lineage.inputs),
            "version": entry.lineage.version,
            "image": entry.lineage.image,
        },
    }


def entry_from_payload(payload: Json) -> models.IndexEntry:
    """Decode a point payload into an entry."""
    lineage = payload["lineage"]
    return models.IndexEntry(
        version=payload["version"],
        chunk_version=payload["chunk_version"],
        chunk_set_version=payload["chunk_set_version"],
        scene_set_version=payload["scene_set_version"],
        asset_version=payload["asset_version"],
        chunking_fingerprint=payload["chunking_fingerprint"],
        segmentation_fingerprint=payload["segmentation_fingerprint"],
        chunk_index=payload["chunk_index"],
        scene_indices=tuple(payload["scene_indices"]),
        start_seconds=payload["start_seconds"],
        end_seconds=payload["end_seconds"],
        text=payload["text"],
        sources=tuple(
            models.SourceView(s["source_id"], s["asset_id"], s["filename"])
            for s in payload["sources"]
        ),
        entitlements=frozenset(payload["entitlements"]),
        classifications=tuple(payload["classifications"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
        lineage=models.Lineage(
            step=lineage["step"],
            fingerprint=lineage["fingerprint"],
            inputs=tuple(lineage["inputs"]),
            version=lineage["version"],
            image=lineage["image"],
        ),
        schema_version=payload["schema_version"],
    )


class QdrantSearchIndex:
    """`SearchIndex` over one Qdrant server."""

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
            raise SearchIndexError(str(exc)) from exc

    def _call(self, method: str, path: str, payload: Json | None = None) -> Any:  # noqa: ANN401 - result shapes vary
        """Exchange and unwrap ``result``; any error status is a `SearchIndexError`."""
        try:
            return self._exchange(method, path, payload).get("result")
        except HttpStatusError as exc:
            raise SearchIndexError(str(exc)) from exc

    def _unusable(self, path: str, exc: Exception) -> SearchIndexError:
        """Name an answer whose shape is not the one the step reads."""
        return SearchIndexError(
            f"{self._base_url}{path} answered something unusable: {type(exc).__name__}: {exc}"
        )

    def version(self) -> str:
        """Return the server version; the connectivity check at startup.

        Raises:
            SearchIndexError: When the server does not answer.
        """
        try:
            answer = self._exchange("GET", "/", None, _PROBE_TIMEOUT_SECONDS)
        except HttpStatusError as exc:
            raise SearchIndexError(str(exc)) from exc
        return str(answer.get("version", ""))

    def describe(self, collection: str) -> models.CollectionInfo | None:
        """Return the collection's dimensions, distance and size; ``None`` when absent."""
        try:
            answer = self._exchange("GET", f"/collections/{collection}")
        except HttpStatusError as exc:
            if exc.status == 404:
                return None
            raise SearchIndexError(str(exc)) from exc
        result = answer.get("result", {})
        try:
            # A dict of named vectors has no "size": the step writes one unnamed vector.
            vectors = result["config"]["params"]["vectors"]
            return models.CollectionInfo(
                dimensions=int(vectors["size"]),
                distance=_DISTANCE_NAMES.get(vectors["distance"], str(vectors["distance"]).lower()),
                point_count=int(result.get("points_count") or 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise self._unusable(f"/collections/{collection}", exc) from exc

    def ensure_collection(self, name: str, dimensions: int, distance: str) -> None:
        """Create the collection with its payload indexes, or verify an existing one."""
        existing = self.describe(name)
        if existing is None:
            try:
                self._call(
                    "PUT",
                    f"/collections/{name}",
                    {"vectors": {"size": dimensions, "distance": _DISTANCES[distance]}},
                )
            except SearchIndexError:
                # Another process may have created it meanwhile: verify instead of failing.
                existing = self.describe(name)
                if existing is None:
                    raise
            else:
                logger.info("created collection %s (%d dims, %s)", name, dimensions, distance)
        if existing is not None and (existing.dimensions, existing.distance) != (
            dimensions,
            distance,
        ):
            raise IndexMismatchError(
                f"collection {name} holds {existing.dimensions}-dimensional {existing.distance} "
                f"vectors, the step produces {dimensions}-dimensional {distance} ones"
            )
        for field in PAYLOAD_INDEXES:
            try:
                self._call(
                    "PUT",
                    f"/collections/{name}/index",
                    {"field_name": field, "field_schema": "keyword"},
                )
            except SearchIndexError as exc:
                # Filtering works without the index, only slower; a refused (re)creation
                # must not keep the step from indexing.
                logger.warning("payload index on %s of %s not ensured: %s", field, name, exc)

    def upsert(self, collection: str, points: Sequence[models.Point]) -> None:
        """Write the points and wait until they are durable."""
        if not points:
            return
        self._call(
            "PUT",
            f"/collections/{collection}/points?wait=true",
            {
                "points": [
                    {"id": p.id, "vector": list(p.vector), "payload": entry_to_payload(p.entry)}
                    for p in points
                ]
            },
        )

    def count(self, collection: str, *, chunk_set_version: str | None = None) -> int:
        """Count the points, exactly."""
        payload: Json = {"exact": True}
        if chunk_set_version is not None:
            payload["filter"] = {
                "must": [{"key": "chunk_set_version", "match": {"value": chunk_set_version}}]
            }
        path = f"/collections/{collection}/points/count"
        result = self._call("POST", path, payload)
        try:
            return int(result["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise self._unusable(path, exc) from exc

    def get_entries(self, collection: str, point_ids: Sequence[str]) -> Sequence[models.IndexEntry]:
        """Return the entries of the points, in the order asked, skipping absent ones."""
        if not point_ids:
            return []
        path = f"/collections/{collection}/points"
        result = self._call(
            "POST", path, {"ids": list(point_ids), "with_payload": True, "with_vector": False}
        )
        try:
            by_id = {str(item["id"]): entry_from_payload(item["payload"]) for item in result}
        except (KeyError, TypeError, ValueError) as exc:
            raise self._unusable(path, exc) from exc
        return [by_id[point_id] for point_id in point_ids if point_id in by_id]

    def drop_collection(self, collection: str) -> None:
        """Delete the collection and every point in it; absent is fine."""
        try:
            self._exchange("DELETE", f"/collections/{collection}")
        except HttpStatusError as exc:
            if exc.status != 404:
                raise SearchIndexError(str(exc)) from exc
