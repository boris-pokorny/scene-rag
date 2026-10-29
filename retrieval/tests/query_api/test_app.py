"""The HTTP adapter over a scripted service: routing, parsing, the header, status codes."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.app import create_app
from scene_rag.retrieval.query_api.caller import ENTITLEMENTS_HEADER
from scene_rag.retrieval.query_api.mcp_server import create_mcp_server
from scene_rag.retrieval.query_api.protocols import (
    IndexUnavailableError,
    InvalidQueryError,
    ModelError,
    SceneApiError,
)
from tests.query_api.test_encoding import ANSWER, SEARCH, sample
from tests.support.query_fakes import TEST_HOSTS, FakeQueryService

HEADERS = {ENTITLEMENTS_HEADER: "public, team-platform"}
CALLER = models.Caller(frozenset({"public", "team-platform"}))


@pytest.fixture
def service() -> FakeQueryService:
    return FakeQueryService(search_result=SEARCH, answer_result=ANSWER)


@pytest.fixture
def client(service: FakeQueryService) -> Iterator[TestClient]:
    app = create_app(service, create_mcp_server(service), allowed_hosts=TEST_HOSTS)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestGivenRoutes:
    """Every route answers from the service with the published encoding."""

    def test_when_health_then_ok_with_served_uncached(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "served": sample("served")}
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"] == "application/json"

    def test_when_searched_then_body_and_header_parsed_and_result_encoded(
        self, client: TestClient, service: FakeQueryService
    ) -> None:
        response = client.post("/search", json=sample("search-request"), headers=HEADERS)

        assert response.status_code == 200
        assert response.json() == sample("search-response")
        assert response.headers["cache-control"] == "no-store"
        assert service.calls == [
            (
                "search",
                models.SearchQuery(
                    question="When does the pipeline cut the video into scenes?",
                    caller=CALLER,
                    top_k=2,
                    filters=models.SearchFilters(asset_version=SEARCH.sources[0].asset_version),
                ),
            )
        ]

    def test_when_answered_then_result_encoded(
        self, client: TestClient, service: FakeQueryService
    ) -> None:
        response = client.post("/answer", json=sample("answer-request"), headers=HEADERS)

        assert response.status_code == 200
        assert response.json() == sample("answer-response")
        assert service.calls == [
            (
                "answer",
                models.AnswerQuery(
                    question="When does the pipeline cut the video into scenes?", caller=CALLER
                ),
            )
        ]

    def test_when_header_absent_then_anonymous_caller(
        self, client: TestClient, service: FakeQueryService
    ) -> None:
        client.post("/search", json={"question": "q"})

        assert service.calls == [("search", models.SearchQuery("q", models.ANONYMOUS))]

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"question": 3},
            {"question": "q", "top_k": "many"},
            {"question": "q", "extra": 1},
            {"question": "q", "filters": {"speaker": "x"}},
        ],
    )
    def test_when_body_invalid_then_400(self, client: TestClient, body: dict[str, object]) -> None:
        response = client.post("/search", json=body, headers=HEADERS)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_parameter"

    def test_when_body_not_json_then_400(self, client: TestClient) -> None:
        response = client.post(
            "/answer", content=b"not json", headers={**HEADERS, "Content-Type": "application/json"}
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_parameter"

    def test_when_path_unknown_then_404_error_body(self, client: TestClient) -> None:
        response = client.get("/nothing")

        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found", "message": "Not Found"}}

    def test_when_method_wrong_then_405_with_allow(self, client: TestClient) -> None:
        response = client.get("/search")

        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"
        assert "POST" in response.headers["allow"]

    def test_when_mcp_route_present_then_served_by_the_same_app(self, client: TestClient) -> None:
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"Accept": "application/json, text/event-stream"},
        )

        assert response.status_code == 200
        assert response.json() == {"jsonrpc": "2.0", "id": 1, "result": {}}


class TestGivenHostHeader:
    """Only the hosts the deployment serves reach the routes and the MCP transport."""

    @pytest.mark.parametrize("path", ["/search", "/mcp"])
    def test_when_host_not_served_then_400_error_body_and_no_call(
        self, service: FakeQueryService, client: TestClient, path: str
    ) -> None:
        response = client.post(
            path, json={"question": "q"}, headers={**HEADERS, "Host": "rebound.example:8081"}
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_parameter"
        assert "rebound.example" in response.json()["error"]["message"]
        assert service.calls == []

    @pytest.mark.parametrize(
        "allowed, host",
        [
            (("localhost",), "LOCALHOST:8081"),
            (("[::1]",), "[::1]:8081"),
            (("127.0.0.1",), "127.0.0.1"),
            (("*",), "anything.example"),
        ],
    )
    def test_when_host_served_then_answered(
        self, service: FakeQueryService, allowed: tuple[str, ...], host: str
    ) -> None:
        app = create_app(service, create_mcp_server(service), allowed_hosts=allowed)

        with TestClient(app) as client:
            response = client.get("/health", headers={"Host": host})

        assert response.status_code == 200


class TestGivenFailingService:
    """Errors the service raises, mapped to status codes and the one error shape."""

    @pytest.mark.parametrize(
        "failure, status, code",
        [
            (InvalidQueryError("question must not be blank"), 400, "invalid_parameter"),
            (IndexUnavailableError("secret://qdrant down"), 503, "index_unavailable"),
            (ModelError("secret://ollama down"), 503, "model_unavailable"),
            (SceneApiError("secret://scene-api down"), 503, "scene_api_unavailable"),
            (RuntimeError("secret://token leaked"), 500, "internal_error"),
        ],
    )
    def test_when_service_raises_then_status_and_code(
        self,
        service: FakeQueryService,
        client: TestClient,
        failure: Exception,
        status: int,
        code: str,
    ) -> None:
        service.failure = failure

        response = client.post("/search", json={"question": "q"}, headers=HEADERS)

        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        assert response.headers["cache-control"] == "no-store"
        assert "secret://" not in response.text

    def test_when_http_exception_raised_then_phrase_coded(self, service: FakeQueryService) -> None:
        app = create_app(service, create_mcp_server(service), allowed_hosts=TEST_HOSTS)

        @app.get("/teapot")
        def teapot() -> None:
            raise HTTPException(418, "short and stout")

        with TestClient(app) as client:
            response = client.get("/teapot")

        assert response.status_code == 418
        assert response.json() == {"error": {"code": "i_m_a_teapot", "message": "short and stout"}}
