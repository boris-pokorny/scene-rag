"""The published OpenAPI document: committed as generated, and true to what the app answers."""

import json
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from scene_rag.extraction.scene_api import models, openapi
from scene_rag.extraction.scene_api.app import create_app
from tests.scene_api.test_app import ASSET, CLIP, RECORDS, SCENE_SET, SUMMARY
from tests.support.scene_api_fakes import FakeSceneCatalog

PUBLISHED = Path(__file__).resolve().parents[2] / "contracts" / "published"
DOCUMENT = json.loads((PUBLISHED / openapi.OUTPUT_NAME).read_text())


def validator_for(reference: str) -> jsonschema.Draft202012Validator:
    """A validator for ``#/components/schemas/<Name>`` resolved within the document."""
    schema = {"$ref": reference, "components": DOCUMENT["components"]}
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


def response_schema(path: str, status: int) -> str:
    responses = DOCUMENT["paths"][path]["get"]["responses"]
    reference: str = responses[str(status)]["content"]["application/json"]["schema"]["$ref"]
    return reference


class TestGivenCommittedDocument:
    """The file under contracts/published."""

    def test_when_regenerated_then_identical(self) -> None:
        assert openapi.generate(PUBLISHED) == (PUBLISHED / openapi.OUTPUT_NAME).read_text()

    def test_when_inspected_then_3_1_get_only_and_no_422(self) -> None:
        assert DOCUMENT["openapi"].startswith("3.1.")
        assert DOCUMENT["info"]["title"] == "scene-rag extraction: scene-api"
        for path, operations in DOCUMENT["paths"].items():
            assert list(operations) == ["get"], path
            assert "422" not in operations["get"]["responses"], path

    def test_when_paths_listed_then_every_route_of_the_app(self) -> None:
        app = create_app(FakeSceneCatalog())
        routes = {route.path for route in app.routes if isinstance(route, Route)}

        assert set(DOCUMENT["paths"]) == routes

    def test_when_references_followed_then_every_component_exists(self) -> None:
        text = json.dumps(DOCUMENT)
        schemas = DOCUMENT["components"]["schemas"]
        for name in ("SceneRecord", "SceneSetSummary", "AssetVersion", "Error", "Health"):
            assert f"#/components/schemas/{name}" in text
            assert name in schemas
        for name, schema in schemas.items():
            assert "$id" not in schema and "$schema" not in schema, name
            assert "#/$defs/" not in json.dumps(schema), name

    def test_when_written_by_main_then_file_matches(self, tmp_path: Path) -> None:
        for file in PUBLISHED.glob("*.schema.json"):
            (tmp_path / file.name).write_text(file.read_text())

        assert openapi.main(["--contracts", str(tmp_path)]) == 0
        assert (tmp_path / openapi.OUTPUT_NAME).read_text() == openapi.generate(PUBLISHED)


class TestGivenAnswers:
    """Every body the app answers validates against the schema the document names for it."""

    @pytest.fixture
    def client(self) -> TestClient:
        catalog = FakeSceneCatalog(
            scene_sets=[SUMMARY],
            records=RECORDS,
            assets=[ASSET],
            derivatives={CLIP.digest: models.DerivativeContent(CLIP, b"wav")},
        )
        return TestClient(create_app(catalog), raise_server_exceptions=False)

    @pytest.mark.parametrize(
        "template, url, status",
        [
            ("/health", "/health", 200),
            ("/scene-sets", "/scene-sets", 200),
            ("/scene-sets", "/scene-sets?limit=0", 400),
            ("/scene-sets/{version}", f"/scene-sets/{SCENE_SET.version}", 200),
            ("/scene-sets/{version}", "/scene-sets/sha256:00", 404),
            ("/scene-sets/{version}/scenes", f"/scene-sets/{SCENE_SET.version}/scenes", 200),
            (
                "/scene-sets/{version}/scenes/{index}",
                f"/scene-sets/{SCENE_SET.version}/scenes/1",
                200,
            ),
            ("/asset-versions/{version}", f"/asset-versions/{ASSET.version}", 200),
            ("/derivatives/{digest}", "/derivatives/sha256:00", 404),
        ],
    )
    def test_when_answered_then_body_matches_documented_schema(
        self, client: TestClient, template: str, url: str, status: int
    ) -> None:
        response = client.get(url)

        assert response.status_code == status
        validator_for(response_schema(template, status)).validate(response.json())

    def test_when_derivative_answered_then_media_type_documented(self, client: TestClient) -> None:
        response = client.get(f"/derivatives/{CLIP.digest}")
        documented = DOCUMENT["paths"]["/derivatives/{digest}"]["get"]["responses"]["200"]

        assert response.headers["content-type"] in documented["content"]
