"""Read model over the scene-api of the extraction service.

`SceneApiClient` is the step's `SceneRecordReader`: it walks the paged listings and decodes
the published documents (scene record, scene set summary, asset version) into the step's own
views. The decoders are this service's copy of the published contract; `tests/contract/`
pins them to sample documents. The HTTP layer lives in `http_transport`.
"""

import http
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any, Final
from urllib.parse import quote, urlencode

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.http_transport import HttpStatusError, JsonTransport
from scene_rag.retrieval.chunking.protocols import SceneApiError

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]

_PROBE_TIMEOUT_SECONDS: Final = 5.0


def _check_schema_version(document: Json, what: str) -> None:
    version = document.get("schema_version")
    if version != models.SCENE_RECORD_SCHEMA_VERSION:
        raise SceneApiError(
            f"{what} has schema_version {version!r}; this step decodes "
            f"{models.SCENE_RECORD_SCHEMA_VERSION}"
        )


def _datetime(text: str) -> datetime:
    return datetime.fromisoformat(text)


def derivative_from_json(document: Json) -> models.DerivativeRefView:
    """Decode a derivative reference."""
    return models.DerivativeRefView(
        digest=document["digest"],
        media_type=document["media_type"],
        size_bytes=document["size_bytes"],
        url=document["url"],
    )


def scene_from_json(document: Json) -> models.SceneView:
    """Decode a scene with its derivative references."""
    audio_clip = document["audio_clip"]
    return models.SceneView(
        index=document["index"],
        start_seconds=document["start_seconds"],
        end_seconds=document["end_seconds"],
        start_frame=document["start_frame"],
        end_frame=document["end_frame"],
        keyframes=tuple(
            models.KeyframeView(
                derivative=derivative_from_json(keyframe),
                frame=keyframe["frame"],
                seconds=keyframe["seconds"],
            )
            for keyframe in document["keyframes"]
        ),
        audio_clip=None if audio_clip is None else derivative_from_json(audio_clip),
    )


def enrichment_from_json(document: Json) -> models.EnrichmentView:
    """Decode one enrichment of a scene record."""
    return models.EnrichmentView(
        version=document["version"],
        enricher=document["enricher"],
        fingerprint=document["fingerprint"],
        classification=document["classification"],
        model=models.ModelRef(name=document["model"]["name"], digest=document["model"]["digest"]),
        segments=tuple(
            models.SegmentView(
                start_seconds=segment["start_seconds"],
                end_seconds=segment["end_seconds"],
                text=segment["text"],
                language=segment["language"],
                confidence=segment["confidence"],
            )
            for segment in document["segments"]
        ),
    )


def asset_from_json(document: Json) -> models.AssetView:
    """Decode what a record says of its asset version."""
    return models.AssetView(
        media_type=document["media_type"],
        size_bytes=document["size_bytes"],
        entitlements=frozenset(document["entitlements"]),
        sources=tuple(
            models.SourceView(
                source_id=source["source_id"],
                asset_id=source["asset_id"],
                filename=source["filename"],
            )
            for source in document["sources"]
        ),
    )


def scene_record_from_json(document: Json) -> models.SceneRecordView:
    """Decode a scene record as ``scene-record.schema.json`` of extraction describes it.

    Raises:
        SceneApiError: When the record's schema version is not the one this step decodes.
    """
    _check_schema_version(document, "scene record")
    media = document["media"]
    return models.SceneRecordView(
        version=document["version"],
        scene_set_version=document["scene_set_version"],
        asset_version=document["asset_version"],
        segmentation_fingerprint=document["segmentation_fingerprint"],
        live=document["live"],
        asset=asset_from_json(document["asset"]),
        media=models.MediaView(
            duration_seconds=media["duration_seconds"],
            frame_rate=media["frame_rate"],
            width=media["width"],
            height=media["height"],
            frame_count=media["frame_count"],
        ),
        scene=scene_from_json(document["scene"]),
        enrichments=tuple(enrichment_from_json(item) for item in document["enrichments"]),
    )


def summary_from_json(document: Json) -> models.SceneSetSummaryView:
    """Decode a scene set summary as ``scene-set-summary.schema.json`` describes it."""
    return models.SceneSetSummaryView(
        version=document["version"],
        asset_version=document["asset_version"],
        segmentation_fingerprint=document["segmentation_fingerprint"],
        scene_count=document["scene_count"],
        created_at=_datetime(document["created_at"]),
        live=document["live"],
        enrichments=tuple(
            models.CoverageView(
                enricher=item["enricher"],
                fingerprint=item["fingerprint"],
                scene_count=item["scene_count"],
                complete=item["complete"],
            )
            for item in document["enrichments"]
        ),
    )


def asset_version_from_json(document: Json) -> models.AssetVersionView:
    """Decode an asset version as ``asset-version.schema.json`` describes it."""
    return models.AssetVersionView(version=document["version"], live=document["live"])


class SceneApiClient:
    """`SceneRecordReader` over the scene-api, paging through every listing it reads."""

    def __init__(self, base_url: str, transport: JsonTransport, timeout_seconds: float) -> None:
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
        """Check that the API answers; the connectivity probe at startup.

        Raises:
            SceneApiError: When it does not, or reports anything but ``ok``.
        """
        answer = self._transport.get(f"{self._base_url}/health", _PROBE_TIMEOUT_SECONDS)
        if answer.get("status") != "ok":
            raise SceneApiError(f"{self._base_url}/health reports {answer!r}")

    def _url(self, path: str, parameters: Sequence[tuple[str, str]] = ()) -> str:
        query = urlencode(parameters)
        return f"{self._base_url}{path}" + (f"?{query}" if query else "")

    def _pages(self, path: str, parameters: Sequence[tuple[str, str]]) -> Iterator[Json]:
        """Yield the items of every page of a listing, following ``next`` until null."""
        after: str | None = None
        while True:
            cursor = [] if after is None else [("after", after)]
            page = self._transport.get(self._url(path, [*parameters, *cursor]), self._timeout)
            items = page.get("items")
            if not isinstance(items, list):
                raise SceneApiError(f"{path} answered no items list")
            yield from items
            after = page.get("next")
            if after is None:
                return
            # Both listings publish `next` as string|null (extraction's page.schema.json); the
            # record listing's cursor is a scene index, but rendered as a string.
            if not isinstance(after, str):
                raise SceneApiError(f"{path} answered a non-string cursor {after!r}")

    def list_scene_sets(self, query: models.SceneSetQuery) -> Iterator[models.SceneSetSummaryView]:
        """Yield the listed scene sets; see `SceneRecordReader.list_scene_sets`."""
        parameters: list[tuple[str, str]] = []
        if query.segmentation_fingerprint is not None:
            parameters.append(("segmentation", query.segmentation_fingerprint))
        parameters.extend(("enrichment", str(pin)) for pin in query.pins)
        for item in self._pages("/scene-sets", parameters):
            yield summary_from_json(item)

    def list_scene_records(
        self, scene_set_version: str, pins: Sequence[models.EnrichmentPin]
    ) -> Sequence[models.SceneRecordView]:
        """Return every scene record of a set; see `SceneRecordReader.list_scene_records`."""
        path = f"/scene-sets/{quote(scene_set_version, safe=':')}/scenes"
        parameters = [("enrichment", str(pin)) for pin in pins]
        return [scene_record_from_json(item) for item in self._pages(path, parameters)]

    def get_asset_version(self, version: str) -> models.AssetVersionView | None:
        """Return the asset version; see `SceneRecordReader.get_asset_version`."""
        url = self._url(f"/asset-versions/{quote(version, safe=':')}")
        try:
            answer = self._transport.get(url, self._timeout)
        except HttpStatusError as exc:
            if exc.status == http.HTTPStatus.NOT_FOUND:
                return None
            raise
        return asset_version_from_json(answer)
