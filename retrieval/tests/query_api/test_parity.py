"""HTTP and MCP answer the same question with the same sources: the evaluation smoke test."""

from collections.abc import Iterator

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp.client import Client

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.app import create_app
from scene_rag.retrieval.query_api.caller import ENTITLEMENTS_HEADER, bind_caller, reset_caller
from scene_rag.retrieval.query_api.mcp_server import create_mcp_server
from tests.support.query_fakes import (
    OTHER_ASSET,
    PUBLIC,
    TEAM,
    TEST_HOSTS,
    QueryHarness,
    make_entry_view,
)

ENTRIES = (
    make_entry_view(0),
    make_entry_view(1),
    make_entry_view(2, asset_version=OTHER_ASSET, entitlements=TEAM, text="team only"),
)
QUESTION = "chunk 1 text"


@pytest.fixture
def query_harness(query_harness: QueryHarness) -> QueryHarness:
    query_harness.seed(*ENTRIES)
    return query_harness


@pytest.fixture
def http(query_harness: QueryHarness) -> Iterator[TestClient]:
    app = create_app(
        query_harness.service, create_mcp_server(query_harness.service), allowed_hosts=TEST_HOSTS
    )
    with TestClient(app) as client:
        yield client


def over_mcp(query_harness: QueryHarness, tool: str, caller: models.Caller) -> object:
    async def main() -> object:
        async with Client(create_mcp_server(query_harness.service)) as client:
            result = await client.call_tool(tool, {"question": QUESTION})
            assert result.is_error is False
            return result.structured_content

    token = bind_caller(caller)
    try:
        return anyio.run(main)
    finally:
        reset_caller(token)


class TestGivenBothAdapters:
    """The same service behind the HTTP routes and the MCP tools."""

    @pytest.mark.parametrize("entitlements", [PUBLIC, TEAM, frozenset()])
    def test_when_searched_then_same_sources_and_served(
        self, query_harness: QueryHarness, http: TestClient, entitlements: frozenset[str]
    ) -> None:
        headers = {ENTITLEMENTS_HEADER: ",".join(sorted(entitlements))}

        over_http = http.post("/search", json={"question": QUESTION}, headers=headers).json()
        mcp = over_mcp(query_harness, "search_scenes", models.Caller(entitlements))

        assert over_http == mcp
        assert over_http["served"]["query_config"] == query_harness.served.query_config
        visible = {e.chunk_version for e in ENTRIES if e.entitlements & entitlements}
        assert {s["chunk_version"] for s in over_http["sources"]} <= visible

    def test_when_answered_then_same_answer_citations_and_sources(
        self, query_harness: QueryHarness, http: TestClient
    ) -> None:
        headers = {ENTITLEMENTS_HEADER: "public"}

        over_http = http.post("/answer", json={"question": QUESTION}, headers=headers).json()
        mcp = over_mcp(query_harness, "answer", models.Caller(PUBLIC))

        assert over_http == mcp
        assert over_http["refused"] is False
        assert over_http["citations"] == [1]
