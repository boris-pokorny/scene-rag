"""HTTP delivery of the scene-api role, over FastAPI: the only module speaking HTTP.

An adapter with no logic of its own: each route parses the request into a query, calls the
`SceneCatalog` it was created over, and encodes the answer. Errors the catalog raises map to
status codes here, and every error body has the one shape of ``error.schema.json``. The
application serves no OpenAPI document of its own: the published one is generated at build
time by `scene_rag.extraction.scene_api.openapi`, with the contract schemas embedded, and the
routes reference its components.
"""

import http
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from scene_rag.extraction.scene_api import encoding, models
from scene_rag.extraction.scene_api.protocols import (
    DerivativeUnavailableError,
    InvalidQueryError,
    NotFoundError,
    SceneCatalog,
)

logger = logging.getLogger(__name__)

TITLE: Final = "scene-rag extraction: scene-api"
API_VERSION: Final = "1.0.0"

NO_STORE: Final = "no-store"
NO_CACHE: Final = "no-cache"
IMMUTABLE: Final = "public, max-age=31536000, immutable"

_FINGERPRINT_PATTERN: Final = r"^[0-9a-f]{12}$"


def _http_code(status: int) -> str:
    """Name an HTTP status the way the error body does: ``not_found``, ``method_not_allowed``."""
    return re.sub(r"[^a-z0-9]+", "_", http.HTTPStatus(status).phrase.lower()).strip("_")


# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]


def _schema(name: str) -> Json:
    return {"$ref": f"#/components/schemas/{name}"}


def _json_response(description: str, schema: str) -> Json:
    return {
        "description": description,
        "content": {"application/json": {"schema": _schema(schema)}},
    }


ERROR_RESPONSES: Final = {
    400: _json_response("A parameter cannot be used as asked.", "Error"),
    404: _json_response("No such resource.", "Error"),
    500: _json_response("The request failed inside the service.", "Error"),
}


def _json(payload: Json, cache: str, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={"Cache-Control": cache})


def _error(
    status: int, code: str, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        encoding.encode_error(code, message),
        status_code=status,
        headers={"Cache-Control": NO_STORE, **(headers or {})},
    )


def etag_of(digest: str) -> str:
    """Return the strong entity tag of a derivative: its digest, quoted."""
    return f'"{digest}"'


def matches_etag(if_none_match: str | None, etag: str) -> bool:
    """Return whether an ``If-None-Match`` header names ``etag`` (or ``*``).

    Weak tags (``W/"..."``) compare equal to their strong form, as RFC 9110 allows for the weak
    comparison ``If-None-Match`` uses.
    """
    if if_none_match is None:
        return False
    candidates = {item.strip().removeprefix("W/") for item in if_none_match.split(",")}
    return "*" in candidates or etag in candidates


def parse_pins(values: Sequence[str] | None) -> tuple[models.EnrichmentPin, ...]:
    """Parse the repeated ``enrichment`` parameter.

    Raises:
        InvalidQueryError: When a value is not ``<enricher>:<fingerprint>``.
    """
    try:
        return tuple(models.EnrichmentPin.parse(value) for value in values or ())
    except ValueError as exc:
        raise InvalidQueryError(str(exc)) from exc


type Pins = Annotated[
    list[str] | None,
    Query(
        description="Enrichment pin `<enricher>:<fingerprint>`; repeatable. With pins, scene "
        "sets are listed only when complete under every pin, and records carry the pinned "
        "enrichments only. Without pins, everything is listed and included.",
    ),
]
type Limit = Annotated[
    int | None,
    Query(
        ge=1, description="Most items on the page; the service's page size by default and at most."
    ),
]


def create_app(catalog: SceneCatalog) -> FastAPI:
    """Build the application over ``catalog``.

    Args:
        catalog: The use cases every route delegates to.

    Returns:
        The application; `bootstrap` serves it, tests drive it with a test client.
    """
    app = FastAPI(title=TITLE, version=API_VERSION, openapi_url=None, docs_url=None, redoc_url=None)

    @app.exception_handler(NotFoundError)
    def not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, "not_found", str(exc))

    @app.exception_handler(InvalidQueryError)
    def invalid_query(request: Request, exc: InvalidQueryError) -> JSONResponse:
        return _error(400, "invalid_parameter", str(exc))

    @app.exception_handler(RequestValidationError)
    def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return _error(400, "invalid_parameter", detail)

    @app.exception_handler(DerivativeUnavailableError)
    def derivative_unavailable(request: Request, exc: DerivativeUnavailableError) -> JSONResponse:
        logger.error("derivative unavailable: %s", exc)
        return _error(503, "derivative_unavailable", str(exc))

    @app.exception_handler(HTTPException)
    def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, _http_code(exc.status_code), str(exc.detail), exc.headers)

    @app.exception_handler(Exception)
    def internal_error(request: Request, exc: Exception) -> JSONResponse:
        # Shapes the body only: Starlette re-raises after this handler, and the server logs the
        # traceback once; logging here would print it twice.
        return _error(500, "internal_error", "the request failed inside the service")

    @app.get(
        "/health",
        summary="Liveness of the process",
        response_model=None,
        responses={200: _json_response("The process answers.", "Health")},
    )
    def health() -> Response:
        return _json({"status": "ok"}, NO_STORE)

    @app.get(
        "/scene-sets",
        summary="List scene sets of live asset versions",
        description="In scene set version order, cursor-paginated: `next` is the version of the "
        "last scene set scanned, so a page may hold fewer items than asked, even none, while "
        "more follow; `next` is null only on the last page.",
        response_model=None,
        responses={
            200: _json_response("A page of scene set summaries.", "PageOfSceneSetSummary"),
            400: ERROR_RESPONSES[400],
            500: ERROR_RESPONSES[500],
        },
    )
    def list_scene_sets(
        asset_version: Annotated[
            str | None, Query(description="Only sets cut from this asset version.")
        ] = None,
        segmentation: Annotated[
            str | None,
            Query(
                pattern=_FINGERPRINT_PATTERN, description="Only sets cut under this fingerprint."
            ),
        ] = None,
        enrichment: Pins = None,
        after: Annotated[str | None, Query(description="The `next` of the previous page.")] = None,
        limit: Limit = None,
    ) -> Response:
        page = catalog.list_scene_sets(
            models.SceneSetQuery(
                asset_version=asset_version,
                segmentation_fingerprint=segmentation,
                pins=parse_pins(enrichment),
                after=after,
                limit=limit,
            )
        )
        return _json(encoding.encode_page(page, encoding.encode_scene_set_summary), NO_CACHE)

    @app.get(
        "/scene-sets/{version}",
        summary="One scene set, live or not",
        response_model=None,
        responses={
            200: _json_response("The scene set's summary.", "SceneSetSummary"),
            404: ERROR_RESPONSES[404],
            500: ERROR_RESPONSES[500],
        },
    )
    def get_scene_set(version: str) -> Response:
        return _json(encoding.encode_scene_set_summary(catalog.get_scene_set(version)), NO_CACHE)

    @app.get(
        "/scene-sets/{version}/scenes",
        summary="List the scene records of a scene set",
        description="In scene index order, cursor-paginated by index: `next` is the index of the "
        "last record on the page, null on the last page.",
        response_model=None,
        responses={
            200: _json_response("A page of scene records.", "PageOfSceneRecord"),
            400: ERROR_RESPONSES[400],
            404: ERROR_RESPONSES[404],
            500: ERROR_RESPONSES[500],
        },
    )
    def list_scene_records(
        version: str,
        enrichment: Pins = None,
        after: Annotated[
            int | None, Query(ge=0, description="The `next` of the previous page: a scene index.")
        ] = None,
        limit: Limit = None,
    ) -> Response:
        page = catalog.list_scene_records(
            version, models.SceneQuery(pins=parse_pins(enrichment), after=after, limit=limit)
        )
        return _json(encoding.encode_page(page, encoding.encode_scene_record), NO_CACHE)

    @app.get(
        "/scene-sets/{version}/scenes/{index}",
        summary="One scene record",
        response_model=None,
        responses={
            200: _json_response("The scene record.", "SceneRecord"),
            400: ERROR_RESPONSES[400],
            404: ERROR_RESPONSES[404],
            500: ERROR_RESPONSES[500],
        },
    )
    def get_scene_record(version: str, index: int, enrichment: Pins = None) -> Response:
        record = catalog.get_scene_record(version, index, parse_pins(enrichment))
        return _json(encoding.encode_scene_record(record), NO_CACHE)

    @app.get(
        "/asset-versions/{version}",
        summary="One asset version, live or retired",
        response_model=None,
        responses={
            200: _json_response("The asset version.", "AssetVersion"),
            404: ERROR_RESPONSES[404],
            500: ERROR_RESPONSES[500],
        },
    )
    def get_asset_version(version: str) -> Response:
        asset = catalog.get_asset_version(version)
        return _json(encoding.encode_asset_version(asset), NO_CACHE)

    @app.get(
        "/derivatives/{digest}",
        summary="The bytes of a keyframe or audio clip",
        description="Content-addressed, so cacheable forever; the ETag is the digest, and a "
        "request whose `If-None-Match` names it is answered 304 without reading the bytes.",
        response_model=None,
        responses={
            200: {
                "description": "The bytes, under the media type the scene set records.",
                "content": {"image/jpeg": {}, "audio/wav": {}},
            },
            304: {"description": "The consumer's copy is current; no body."},
            404: ERROR_RESPONSES[404],
            500: ERROR_RESPONSES[500],
            503: _json_response("The bytes cannot be read as filed.", "Error"),
        },
    )
    def get_derivative(request: Request, digest: str) -> Response:
        reference = catalog.find_derivative(digest)
        headers = {"ETag": etag_of(reference.digest), "Cache-Control": IMMUTABLE}
        if matches_etag(request.headers.get("if-none-match"), headers["ETag"]):
            return Response(status_code=304, headers=headers)
        content = catalog.get_derivative(digest)
        return Response(content.data, media_type=reference.media_type, headers=headers)

    return app
