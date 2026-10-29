"""HTTP delivery of the query-api role, over FastAPI: the only module serving HTTP.

An adapter with no logic of its own: each route parses the request into a query, calls the
`QueryService` it was created over as the caller the middleware bound, and encodes the
answer. Errors the service raises map to status codes here, and every error body has the one
shape of ``error.schema.json``. The MCP server is served by the same application under
`MCP_PATH`, so one process, one port and one trusted header serve both adapters; a request
naming a host the deployment does not serve reaches neither. The
application serves no OpenAPI document of its own: the published one is generated at build
time by `scene_rag.retrieval.query_api.openapi`, with the contract schemas embedded.
"""

import contextlib
import http
import logging
import re
from collections.abc import AsyncIterator, Collection, Mapping
from typing import Any, Final

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from scene_rag.retrieval.query_api import encoding, models
from scene_rag.retrieval.query_api.caller import CallerMiddleware, current_caller
from scene_rag.retrieval.query_api.protocols import (
    IndexUnavailableError,
    InvalidQueryError,
    ModelError,
    NotFoundError,
    QueryService,
    SceneApiError,
)

logger = logging.getLogger(__name__)

MCP_PATH: Final = "/mcp"
NO_STORE: Final = "no-store"

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]


def _http_code(status: int) -> str:
    """Name an HTTP status the way the error body does: ``not_found``, ``method_not_allowed``."""
    return re.sub(r"[^a-z0-9]+", "_", http.HTTPStatus(status).phrase.lower()).strip("_")


def _schema(name: str) -> Json:
    return {"$ref": f"#/components/schemas/{name}"}


def _json_response(description: str, schema: str) -> Json:
    return {
        "description": description,
        "content": {"application/json": {"schema": _schema(schema)}},
    }


ERROR_RESPONSES: Final = {
    400: _json_response("The request cannot be served as asked.", "Error"),
    404: _json_response("No such resource.", "Error"),
    500: _json_response("The request failed inside the service.", "Error"),
    503: _json_response("A backend the request needs does not answer.", "Error"),
}


def _json(payload: Mapping[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(dict(payload), status_code=status, headers={"Cache-Control": NO_STORE})


def _error(
    status: int, code: str, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        dict(encoding.encode_error(code, message)),
        status_code=status,
        headers={"Cache-Control": NO_STORE, **(headers or {})},
    )


def _host_of(value: str) -> str:
    """Return the Host header value without its port: ``[::1]:8081`` -> ``[::1]``."""
    if value.startswith("["):
        return value[: value.find("]") + 1].lower()
    return value.split(":", 1)[0].lower()


class TrustedHostMiddleware:
    """ASGI middleware refusing a request whose Host header names a host not served.

    The entitlements header is trusted, so a page whose name was rebound to this address
    must not reach the routes: its requests carry its own name as the host. Starlette's
    middleware of the same name answers plain text; this one answers the error document.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: Collection[str]) -> None:
        """Wrap ``app``, serving ``allowed_hosts``; ``*`` among them serves every host."""
        self._app = app
        self._allowed = frozenset(host.lower() for host in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Refuse a request to a host not served, pass every other one through."""
        if scope["type"] == "http" and "*" not in self._allowed:
            value = next(
                (raw.decode("latin-1") for name, raw in scope["headers"] if name == b"host"), ""
            )
            if _host_of(value) not in self._allowed:
                response = _error(400, "invalid_parameter", f"host {value!r} is not served here")
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


class FiltersRequest(BaseModel):
    """What a request may narrow the search by."""

    model_config = ConfigDict(extra="forbid")

    asset_version: str | None = None


class QueryRequest(BaseModel):
    """The body of ``POST /search`` and ``POST /answer``."""

    model_config = ConfigDict(extra="forbid")

    question: str
    top_k: int | None = None
    filters: FiltersRequest = FiltersRequest()

    def filters_model(self) -> models.SearchFilters:
        """Return the filters as the service takes them."""
        return models.SearchFilters(asset_version=self.filters.asset_version)


def create_app(
    service: QueryService, mcp_server: MCPServer, *, allowed_hosts: Collection[str]
) -> FastAPI:
    """Build the application over ``service``, serving ``mcp_server`` under `MCP_PATH`.

    Args:
        service: The use cases every route delegates to.
        mcp_server: The MCP adapter over the same service; its streamable HTTP transport is
            stateless and answers JSON, so each call runs in its request's task and sees the
            caller the middleware bound.
        allowed_hosts: Host header values served, without the port; ``*`` serves every
            host. Checked for the REST routes and the MCP transport alike.

    Returns:
        The application; `bootstrap` serves it, tests drive it with a test client.
    """
    mcp_app = mcp_server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        stateless_http=True,
        json_response=True,
        # `TrustedHostMiddleware` checks the host for every route, this one included; the
        # SDK's own check would need the same list a second time.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with mcp_server.session_manager.run():
            yield

    app = FastAPI(
        title=encoding.TITLE,
        version=encoding.API_VERSION,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(CallerMiddleware)
    # Added last, so it runs first: a refused request binds no caller.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

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

    @app.exception_handler(NotFoundError)
    def not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, "not_found", str(exc))

    @app.exception_handler(IndexUnavailableError)
    def index_unavailable(request: Request, exc: IndexUnavailableError) -> JSONResponse:
        logger.error("index unavailable: %s", exc)
        return _error(503, "index_unavailable", encoding.BACKEND_MESSAGES["index_unavailable"])

    @app.exception_handler(ModelError)
    def model_unavailable(request: Request, exc: ModelError) -> JSONResponse:
        logger.error("model unavailable: %s", exc)
        return _error(503, "model_unavailable", encoding.BACKEND_MESSAGES["model_unavailable"])

    @app.exception_handler(SceneApiError)
    def scene_api_unavailable(request: Request, exc: SceneApiError) -> JSONResponse:
        logger.error("scene-api unavailable: %s", exc)
        return _error(
            503, "scene_api_unavailable", encoding.BACKEND_MESSAGES["scene_api_unavailable"]
        )

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
        summary="Liveness of the process, and what it serves",
        response_model=None,
        responses={200: _json_response("The process answers.", "Health")},
    )
    def health() -> Response:
        return _json(encoding.encode_health(service.served))

    @app.post(
        "/search",
        summary="The scenes relevant to a question",
        description="Sources the caller may see, best first. The caller's entitlements come "
        "from the `X-Scene-Rag-Entitlements` header; without it nothing is visible. `top_k` "
        "defaults to and is capped by the served configuration.",
        response_model=None,
        responses={
            200: _json_response("The relevant sources.", "SearchResponse"),
            400: ERROR_RESPONSES[400],
            500: ERROR_RESPONSES[500],
            503: ERROR_RESPONSES[503],
        },
    )
    def search(body: QueryRequest) -> Response:
        result = service.search(
            models.SearchQuery(
                question=body.question,
                caller=current_caller(),
                top_k=body.top_k,
                filters=body.filters_model(),
            )
        )
        return _json(encoding.encode_search_result(result))

    @app.post(
        "/answer",
        summary="An answer written over the relevant scenes, with citations",
        description="`[n]` in the answer cites `sources[n-1]`; `citations` lists the numbers "
        "that name a source, `invalid_citations` those that do not. A refusal carries no "
        "citations and consumed no tokens.",
        response_model=None,
        responses={
            200: _json_response("The answer, or a refusal.", "AnswerResponse"),
            400: ERROR_RESPONSES[400],
            500: ERROR_RESPONSES[500],
            503: ERROR_RESPONSES[503],
        },
    )
    def answer(body: QueryRequest) -> Response:
        result = service.answer(
            models.AnswerQuery(
                question=body.question,
                caller=current_caller(),
                top_k=body.top_k,
                filters=body.filters_model(),
            )
        )
        return _json(encoding.encode_answer_result(result))

    # The MCP transport is one more route of this application, not a mounted sub-application:
    # no path prefix doubling, no redirect, and the caller middleware above it.
    app.router.routes.extend(mcp_app.routes)
    return app
