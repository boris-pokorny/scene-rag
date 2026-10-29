"""The MCP adapter over a scripted service: tools, resources, errors, the caller."""

import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import httpx2
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import (
    BlobResourceContents,
    CallToolResult,
    ReadResourceResult,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
)

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.app import create_app
from scene_rag.retrieval.query_api.caller import ENTITLEMENTS_HEADER, bind_caller, reset_caller
from scene_rag.retrieval.query_api.mcp_server import (
    KEYFRAME_URI,
    SCENE_URI,
    SERVER_NAME,
    create_mcp_server,
)
from scene_rag.retrieval.query_api.protocols import IndexUnavailableError
from tests.query_api.test_encoding import ANSWER, SEARCH, sample
from tests.support.query_fakes import FakeQueryService

CALLER = models.Caller(frozenset({"public", "team-platform"}))
SET = "sha256:" + "37" * 32
RECORD: dict[str, Any] = {"schema_version": 1, "asset_version": "sha256:aa", "scene": {"index": 2}}
DIGEST = "sha256:" + "6a" * 32


def run[T](main: Callable[[], Awaitable[T]]) -> T:
    return anyio.run(main)


@pytest.fixture
def service() -> FakeQueryService:
    return FakeQueryService(
        search_result=SEARCH,
        answer_result=ANSWER,
        records={(SET, 2): RECORD},
        derivatives={DIGEST: models.DerivativeContent("image/jpeg", b"\xff\xd8jpeg")},
    )


class TestGivenInProcessClient:
    """The SDK client connected to the server directly; the caller bound as over stdio."""

    @pytest.fixture(autouse=True)
    def bound_caller(self) -> Any:  # noqa: ANN401 - a contextvar token
        token = bind_caller(CALLER)
        yield token
        reset_caller(token)

    def test_when_tools_listed_then_two_read_only_tools_with_schemas(
        self, service: FakeQueryService
    ) -> None:
        async def main() -> list[Tool]:
            async with Client(create_mcp_server(service)) as client:
                return (await client.list_tools()).tools

        tools = run(main)

        assert [t.name for t in tools] == ["search_scenes", "answer"]
        for tool in tools:
            assert tool.annotations is not None and tool.annotations.read_only_hint is True
            assert tool.input_schema["required"] == ["question"]
            assert tool.output_schema is not None and "served" in tool.output_schema["required"]

    def test_when_search_called_then_structured_output_is_the_published_document(
        self, service: FakeQueryService
    ) -> None:
        async def main() -> CallToolResult:
            async with Client(create_mcp_server(service)) as client:
                return await client.call_tool(
                    "search_scenes", {"question": "q", "top_k": 2, "asset_version": "sha256:aa"}
                )

        result = run(main)

        assert result.is_error is False
        assert result.structured_content == sample("search-response")
        assert service.calls == [
            (
                "search",
                models.SearchQuery(
                    "q", CALLER, top_k=2, filters=models.SearchFilters(asset_version="sha256:aa")
                ),
            )
        ]

    def test_when_answer_called_then_structured_output_is_the_published_document(
        self, service: FakeQueryService
    ) -> None:
        async def main() -> CallToolResult:
            async with Client(create_mcp_server(service)) as client:
                return await client.call_tool("answer", {"question": "q"})

        result = run(main)

        assert result.is_error is False
        assert result.structured_content == sample("answer-response")
        assert service.calls == [("answer", models.AnswerQuery("q", CALLER))]

    def test_when_service_fails_then_tool_error_names_the_code(
        self, service: FakeQueryService
    ) -> None:
        service.failure = IndexUnavailableError("qdrant down")

        async def main() -> CallToolResult:
            async with Client(create_mcp_server(service)) as client:
                return await client.call_tool("search_scenes", {"question": "q"})

        result = run(main)

        assert result.is_error is True
        assert isinstance(result.content[0], TextContent)
        assert "index_unavailable: the search index cannot be read" in result.content[0].text
        assert "qdrant down" not in result.content[0].text

    def test_when_arguments_invalid_then_tool_error_not_service_call(
        self, service: FakeQueryService
    ) -> None:
        async def main() -> CallToolResult:
            async with Client(create_mcp_server(service)) as client:
                return await client.call_tool("search_scenes", {"top_k": 2})

        result = run(main)

        assert result.is_error is True
        assert service.calls == []

    def test_when_templates_listed_then_scene_and_keyframe(self, service: FakeQueryService) -> None:
        async def main() -> list[ResourceTemplate]:
            async with Client(create_mcp_server(service)) as client:
                return (await client.list_resource_templates()).resource_templates

        templates = run(main)

        assert [t.uri_template for t in templates] == [SCENE_URI, KEYFRAME_URI]
        assert [t.mime_type for t in templates] == ["application/json", "image/jpeg"]

    def test_when_scene_read_then_record_as_json_for_the_caller(
        self, service: FakeQueryService
    ) -> None:
        async def main() -> ReadResourceResult:
            async with Client(create_mcp_server(service)) as client:
                return await client.read_resource(f"scene://scene-sets/{SET}/2")

        result = run(main)

        (content,) = result.contents
        assert isinstance(content, TextResourceContents)
        assert json.loads(content.text) == RECORD
        assert content.mime_type == "application/json"
        assert service.calls == [("get_scene", SET, 2, CALLER)]

    def test_when_keyframe_read_then_blob(self, service: FakeQueryService) -> None:
        async def main() -> ReadResourceResult:
            async with Client(create_mcp_server(service)) as client:
                return await client.read_resource(f"keyframe://derivatives/{DIGEST}")

        result = run(main)

        (content,) = result.contents
        assert isinstance(content, BlobResourceContents)
        assert base64.b64decode(content.blob) == b"\xff\xd8jpeg"
        assert content.mime_type == "image/jpeg"
        assert service.calls == [("get_keyframe", DIGEST, CALLER)]

    def test_when_scene_missing_then_resource_error_names_the_code(
        self, service: FakeQueryService
    ) -> None:
        async def main() -> str:
            async with Client(create_mcp_server(service)) as client:
                try:
                    await client.read_resource(f"scene://scene-sets/{SET}/9")
                except Exception as exc:  # the SDK's protocol error; its message is the point
                    return str(exc)
                return ""

        assert f"not_found: no scene {SET}/9" in run(main)

    def test_when_server_described_then_named_and_instructed(
        self, service: FakeQueryService
    ) -> None:
        server = create_mcp_server(service)

        assert server.name == SERVER_NAME
        assert server.instructions is not None and ENTITLEMENTS_HEADER in server.instructions


class TestGivenHttpTransport:
    """The server behind the application's /mcp route; the caller from the header."""

    def test_when_called_over_http_then_header_becomes_the_caller(
        self, service: FakeQueryService
    ) -> None:
        app = create_app(service, create_mcp_server(service), allowed_hosts=("query-api",))

        async def main() -> list[CallToolResult]:
            async with app.router.lifespan_context(app):
                http = httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app),
                    base_url="http://query-api:8081",
                    headers={ENTITLEMENTS_HEADER: "public"},
                )
                transport = streamable_http_client("http://query-api:8081/mcp", http_client=http)
                async with http, Client(transport) as client:
                    first = await client.call_tool("search_scenes", {"question": "q"})
                    second = await client.call_tool("search_scenes", {"question": "r"})
                    return [first, second]

        first, second = run(main)

        assert first.is_error is False and second.is_error is False
        assert first.structured_content == sample("search-response")
        assert service.calls == [
            ("search", models.SearchQuery("q", models.Caller(frozenset({"public"})))),
            ("search", models.SearchQuery("r", models.Caller(frozenset({"public"})))),
        ]

    def test_when_called_without_header_then_anonymous(self, service: FakeQueryService) -> None:
        app = create_app(service, create_mcp_server(service), allowed_hosts=("query-api",))

        async def main() -> CallToolResult:
            async with app.router.lifespan_context(app):
                http = httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app), base_url="http://query-api:8081"
                )
                transport = streamable_http_client("http://query-api:8081/mcp", http_client=http)
                async with http, Client(transport) as client:
                    return await client.call_tool("answer", {"question": "q"})

        result = run(main)

        assert result.is_error is False
        assert service.calls == [("answer", models.AnswerQuery("q", models.ANONYMOUS))]
