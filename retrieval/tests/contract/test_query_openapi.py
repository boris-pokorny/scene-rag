"""The published OpenAPI document: committed as generated, and true to what the app answers."""

import json
from collections.abc import Iterator
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from scene_rag.retrieval.query_api import openapi
from scene_rag.retrieval.query_api.app import MCP_PATH, create_app
from scene_rag.retrieval.query_api.caller import ENTITLEMENTS_HEADER
from scene_rag.retrieval.query_api.mcp_server import create_mcp_server
from scene_rag.retrieval.query_api.protocols import IndexUnavailableError
from tests.query_api.test_encoding import ANSWER, SEARCH
from tests.support.query_fakes import TEST_HOSTS, FakeQueryService

PUBLISHED = Path(__file__).resolve().parents[2] / "contracts" / "published"
DOCUMENT = json.loads((PUBLISHED / openapi.OUTPUT_NAME).read_text())


def validator_for(reference: str) -> jsonschema.Draft202012Validator:
    """A validator for ``#/components/schemas/<Name>`` resolved within the document."""
    schema = {"$ref": reference, "components": DOCUMENT["components"]}
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


def response_schema(path: str, method: str, status: int) -> str:
    responses = DOCUMENT["paths"][path][method]["responses"]
    reference: str = responses[str(status)]["content"]["application/json"]["schema"]["$ref"]
    return reference


class TestGivenCommittedDocument:
    """The file under contracts/published."""

    def test_when_regenerated_then_identical(self) -> None:
        assert openapi.generate(PUBLISHED) == (PUBLISHED / openapi.OUTPUT_NAME).read_text()

    def test_when_inspected_then_3_1_and_no_422(self) -> None:
        assert DOCUMENT["openapi"].startswith("3.1.")
        assert DOCUMENT["info"]["title"] == "scene-rag retrieval: query-api"
        for path, operations in DOCUMENT["paths"].items():
            for method, operation in operations.items():
                assert "422" not in operation["responses"], (path, method)

    def test_when_paths_listed_then_every_route_of_the_app_but_mcp(self) -> None:
        service = FakeQueryService()
        app = create_app(service, create_mcp_server(service), allowed_hosts=TEST_HOSTS)
        routes = {route.path for route in app.routes if isinstance(route, Route)}

        assert set(DOCUMENT["paths"]) == routes - {MCP_PATH}
        assert MCP_PATH in routes

    def test_when_request_bodies_inspected_then_published_schemas(self) -> None:
        for path, name in (("/search", "SearchRequest"), ("/answer", "AnswerRequest")):
            body = DOCUMENT["paths"][path]["post"]["requestBody"]
            assert body["required"] is True
            assert body["content"]["application/json"]["schema"] == {
                "$ref": f"#/components/schemas/{name}"
            }

    def test_when_references_followed_then_every_component_exists(self) -> None:
        text = json.dumps(DOCUMENT)
        schemas = DOCUMENT["components"]["schemas"]
        for name in openapi.SCHEMA_FILES:
            assert f"#/components/schemas/{name}" in text
            assert name in schemas
        for name, schema in schemas.items():
            assert "$id" not in schema and "$schema" not in schema, name
            assert "#/$defs/" not in json.dumps(schema), name
            assert openapi.SCHEMA_ID_PREFIX not in json.dumps(schema), name

    def test_when_written_by_main_then_file_matches(self, tmp_path: Path) -> None:
        for file in PUBLISHED.glob("*.schema.json"):
            (tmp_path / file.name).write_text(file.read_text())

        assert openapi.main(["--contracts", str(tmp_path)]) == 0
        assert (tmp_path / openapi.OUTPUT_NAME).read_text() == openapi.generate(PUBLISHED)


class TestGivenAnswers:
    """Every body the app answers validates against the schema the document names for it."""

    @pytest.fixture
    def service(self) -> FakeQueryService:
        return FakeQueryService(search_result=SEARCH, answer_result=ANSWER)

    @pytest.fixture
    def client(self, service: FakeQueryService) -> Iterator[TestClient]:
        app = create_app(service, create_mcp_server(service), allowed_hosts=TEST_HOSTS)
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    @pytest.mark.parametrize(
        "path, method, body, status",
        [
            ("/health", "get", None, 200),
            ("/search", "post", {"question": "q"}, 200),
            ("/answer", "post", {"question": "q", "top_k": 3}, 200),
            ("/search", "post", {"question": 1}, 400),
            ("/answer", "post", {}, 400),
        ],
    )
    def test_when_answered_then_body_matches_documented_schema(
        self,
        client: TestClient,
        path: str,
        method: str,
        body: dict[str, object] | None,
        status: int,
    ) -> None:
        response = client.request(
            method.upper(), path, json=body, headers={ENTITLEMENTS_HEADER: "public"}
        )

        assert response.status_code == status
        validator_for(response_schema(path, method, status)).validate(response.json())

    def test_when_backend_down_then_503_matches_documented_schema(
        self, service: FakeQueryService, client: TestClient
    ) -> None:
        service.failure = IndexUnavailableError("down")

        response = client.post("/search", json={"question": "q"})

        assert response.status_code == 503
        validator_for(response_schema("/search", "post", 503)).validate(response.json())
