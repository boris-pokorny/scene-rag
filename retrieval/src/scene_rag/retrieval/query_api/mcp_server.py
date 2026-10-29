"""MCP delivery of the query-api role, over the official SDK: tools and resources.

An adapter with no logic of its own: each tool calls the `QueryService` it was created
over as the caller the transport bound (the trusted header over HTTP, the command line over
stdio) and returns the same document the HTTP adapter publishes, declared as structured
output. The resources proxy scene records and keyframes. The server is read-only. Errors the
service raises become tool or resource errors carrying the HTTP error code and message, so a
client sees a failed call, never a broken protocol.
"""

import contextlib
import json
import logging
from collections.abc import Iterator
from typing import Final

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp_types import ToolAnnotations

from scene_rag.retrieval.query_api import encoding, models
from scene_rag.retrieval.query_api.caller import ENTITLEMENTS_HEADER, current_caller
from scene_rag.retrieval.query_api.protocols import (
    IndexUnavailableError,
    InvalidQueryError,
    ModelError,
    NotFoundError,
    QueryService,
    SceneApiError,
)

logger = logging.getLogger(__name__)

SERVER_NAME: Final = "scene-rag-query-api"
SCENE_URI: Final = "scene://scene-sets/{scene_set_version}/{index}"
KEYFRAME_URI: Final = "keyframe://derivatives/{digest}"
INSTRUCTIONS: Final = (
    "Read-only search over a video library cut into scenes. `search_scenes` returns the "
    "scene spans relevant to a question, best first, with timestamps and provenance; write "
    "the answer from them and cite them. `answer` returns an answer written by the server's "
    "own model with `[n]` citations into its `sources`. A source's `scene_set_version` and "
    "`scene_indices` name `scene://` resources; a scene record names `keyframe://` "
    f"resources. Over HTTP the caller's entitlements come from the `{ENTITLEMENTS_HEADER}` "
    "header; without them nothing is visible."
)

_ERROR_CODES: Final[tuple[tuple[type[Exception], str], ...]] = (
    (InvalidQueryError, "invalid_parameter"),
    (NotFoundError, "not_found"),
    (IndexUnavailableError, "index_unavailable"),
    (ModelError, "model_unavailable"),
    (SceneApiError, "scene_api_unavailable"),
)


def _code_of(exc: Exception) -> str | None:
    for kind, code in _ERROR_CODES:
        if isinstance(exc, kind):
            return code
    return None


def _message_of(code: str, exc: Exception) -> str:
    """Return what the client is told: a backend failure is logged and named generically."""
    message = encoding.BACKEND_MESSAGES.get(code)
    if message is None:
        return str(exc)
    logger.error("%s: %s", code.replace("_", " "), exc)
    return message


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn a domain error into a `ToolError` naming the HTTP error code."""
    try:
        yield
    except Exception as exc:
        code = _code_of(exc)
        if code is None:
            raise
        raise ToolError(f"{code}: {_message_of(code, exc)}") from exc


@contextlib.contextmanager
def _resource_errors() -> Iterator[None]:
    """Turn a domain error into a `ResourceError` naming the HTTP error code."""
    try:
        yield
    except Exception as exc:
        code = _code_of(exc)
        if code is None:
            raise
        raise ResourceError(f"{code}: {_message_of(code, exc)}") from exc


def create_mcp_server(service: QueryService) -> MCPServer:
    """Build the MCP server over ``service``.

    Args:
        service: The use cases every tool and resource delegates to.

    Returns:
        The server; `app.create_app` serves its streamable HTTP transport, `bootstrap` runs
        it over stdio for local use.
    """
    server = MCPServer(
        name=SERVER_NAME,
        version=encoding.API_VERSION,
        instructions=INSTRUCTIONS,
    )
    read_only = ToolAnnotations(read_only_hint=True)

    @server.tool(
        name="search_scenes",
        title="Search scenes",
        description="The scene spans relevant to a question, best first, that the caller may "
        "see. `top_k` defaults to and is capped by the served configuration; `asset_version` "
        "narrows the search to one asset version.",
        annotations=read_only,
    )
    def search_scenes(
        question: str, top_k: int | None = None, asset_version: str | None = None
    ) -> encoding.SearchResponseDocument:
        with _tool_errors():
            result = service.search(
                models.SearchQuery(
                    question=question,
                    caller=current_caller(),
                    top_k=top_k,
                    filters=models.SearchFilters(asset_version=asset_version),
                )
            )
        return encoding.encode_search_result(result)

    @server.tool(
        name="answer",
        title="Answer a question",
        description="An answer written over the relevant scene spans by the server's own "
        "model, or a refusal. `[n]` in the answer cites `sources[n-1]`.",
        annotations=read_only,
    )
    def answer(
        question: str, top_k: int | None = None, asset_version: str | None = None
    ) -> encoding.AnswerResponseDocument:
        with _tool_errors():
            result = service.answer(
                models.AnswerQuery(
                    question=question,
                    caller=current_caller(),
                    top_k=top_k,
                    filters=models.SearchFilters(asset_version=asset_version),
                )
            )
        return encoding.encode_answer_result(result)

    @server.resource(
        SCENE_URI,
        name="scene",
        title="Scene record",
        description="One scene with its enrichments, keyframes and audio clip, as the "
        "extraction service publishes it; visible only to a caller entitled to its asset.",
        mime_type="application/json",
    )
    def scene(scene_set_version: str, index: int) -> str:
        with _resource_errors():
            record = service.get_scene(scene_set_version, index, current_caller())
        return json.dumps(record, sort_keys=True)

    @server.resource(
        KEYFRAME_URI,
        name="keyframe",
        title="Keyframe",
        description="The JPEG of a keyframe, by the content digest a scene record names.",
        mime_type="image/jpeg",
    )
    def keyframe(digest: str) -> bytes:
        with _resource_errors():
            return service.get_keyframe(digest, current_caller()).data

    return server
