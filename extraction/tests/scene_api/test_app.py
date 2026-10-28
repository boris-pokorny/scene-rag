"""The HTTP adapter over a scripted catalog: routing, parsing, status codes, headers."""

import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from scene_rag.extraction.scene_api import models
from scene_rag.extraction.scene_api.app import create_app, matches_etag, parse_pins
from scene_rag.extraction.scene_api.encoding import (
    encode_asset_version,
    encode_scene_record,
    encode_scene_set_summary,
)
from scene_rag.extraction.scene_api.protocols import DerivativeUnavailableError, InvalidQueryError
from tests.support.scene_api_fakes import (
    T0,
    TRANSCRIPT_FP,
    FakeSceneCatalog,
    clip_of,
    make_asset_view,
    make_record,
    make_scene_set_view,
)

SCENE_SET = make_scene_set_view()
RECORDS = tuple(make_record(SCENE_SET, index) for index in range(3))
SUMMARY = models.SceneSetSummary(
    version=SCENE_SET.version,
    asset_version=SCENE_SET.asset_version,
    segmentation_fingerprint=SCENE_SET.segmentation_fingerprint,
    scene_count=3,
    created_at=T0,
    live=True,
    enrichments=(models.EnrichmentCoverage("transcript", TRANSCRIPT_FP, 3, True),),
)
ASSET = make_asset_view(SCENE_SET.asset_version)
CLIP = clip_of(SCENE_SET)
PIN = f"transcript:{TRANSCRIPT_FP}"


@pytest.fixture
def catalog() -> FakeSceneCatalog:
    return FakeSceneCatalog(
        scene_sets=[SUMMARY],
        records=RECORDS,
        assets=[ASSET],
        derivatives={CLIP.digest: models.DerivativeContent(CLIP, b"wav")},
    )


@pytest.fixture
def client(catalog: FakeSceneCatalog) -> TestClient:
    return TestClient(create_app(catalog), raise_server_exceptions=False)


class TestGivenRoutes:
    """Every route answers from the catalog with the published encoding."""

    def test_when_health_then_ok_uncached(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"] == "application/json"

    def test_when_scene_sets_listed_then_query_parsed_and_page_encoded(
        self, client: TestClient, catalog: FakeSceneCatalog
    ) -> None:
        response = client.get(
            "/scene-sets",
            params=[
                ("asset_version", "sha256:ab"),
                ("segmentation", "0123456789ab"),
                ("enrichment", PIN),
                ("enrichment", "description:3b9e1f6a2c40"),
                ("after", "sha256:00"),
                ("limit", "5"),
            ],
        )

        assert response.status_code == 200
        assert response.json() == {"items": [encode_scene_set_summary(SUMMARY)], "next": "cursor-1"}
        assert response.headers["cache-control"] == "no-cache"
        assert catalog.calls == [
            (
                "list_scene_sets",
                models.SceneSetQuery(
                    asset_version="sha256:ab",
                    segmentation_fingerprint="0123456789ab",
                    pins=(
                        models.EnrichmentPin("transcript", TRANSCRIPT_FP),
                        models.EnrichmentPin("description", "3b9e1f6a2c40"),
                    ),
                    after="sha256:00",
                    limit=5,
                ),
            )
        ]

    def test_when_scene_sets_listed_bare_then_defaults(
        self, client: TestClient, catalog: FakeSceneCatalog
    ) -> None:
        assert client.get("/scene-sets").status_code == 200
        assert catalog.calls == [("list_scene_sets", models.SceneSetQuery())]

    def test_when_scene_set_fetched_then_summary(self, client: TestClient) -> None:
        response = client.get(f"/scene-sets/{SCENE_SET.version}")

        assert response.status_code == 200
        assert response.json() == encode_scene_set_summary(SUMMARY)

    def test_when_records_listed_then_query_parsed(
        self, client: TestClient, catalog: FakeSceneCatalog
    ) -> None:
        response = client.get(
            f"/scene-sets/{SCENE_SET.version}/scenes",
            params={"after": "1", "limit": "2", "enrichment": PIN},
        )

        assert response.status_code == 200
        assert response.json() == {"items": [encode_scene_record(r) for r in RECORDS], "next": None}
        assert catalog.calls == [
            (
                "list_scene_records",
                (
                    SCENE_SET.version,
                    models.SceneQuery(
                        pins=(models.EnrichmentPin("transcript", TRANSCRIPT_FP),), after=1, limit=2
                    ),
                ),
            )
        ]

    def test_when_record_fetched_then_pins_passed(
        self, client: TestClient, catalog: FakeSceneCatalog
    ) -> None:
        response = client.get(
            f"/scene-sets/{SCENE_SET.version}/scenes/2", params={"enrichment": PIN}
        )

        assert response.status_code == 200
        assert response.json() == encode_scene_record(RECORDS[2])
        assert catalog.calls == [
            (
                "get_scene_record",
                (SCENE_SET.version, 2, (models.EnrichmentPin("transcript", TRANSCRIPT_FP),)),
            )
        ]

    def test_when_asset_fetched_then_encoded(self, client: TestClient) -> None:
        response = client.get(f"/asset-versions/{ASSET.version}")

        assert response.status_code == 200
        assert response.json() == encode_asset_version(ASSET)

    def test_when_derivative_fetched_then_bytes_immutable_with_etag(
        self, client: TestClient
    ) -> None:
        response = client.get(f"/derivatives/{CLIP.digest}")

        assert response.status_code == 200
        assert response.content == b"wav"
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["content-length"] == "3"
        assert response.headers["etag"] == f'"{CLIP.digest}"'
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"

    @pytest.mark.parametrize(
        "header",
        [f'"{CLIP.digest}"', f'W/"{CLIP.digest}"', f'"sha256:other", "{CLIP.digest}"', "*"],
    )
    def test_when_derivative_revalidated_with_its_etag_then_304_unread(
        self, catalog: FakeSceneCatalog, client: TestClient, header: str
    ) -> None:
        response = client.get(f"/derivatives/{CLIP.digest}", headers={"If-None-Match": header})

        assert response.status_code == 304
        assert response.content == b""
        assert response.headers["etag"] == f'"{CLIP.digest}"'
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert [name for name, _ in catalog.calls] == ["find_derivative"]

    def test_when_derivative_revalidated_with_another_etag_then_bytes(
        self, catalog: FakeSceneCatalog, client: TestClient
    ) -> None:
        response = client.get(
            f"/derivatives/{CLIP.digest}", headers={"If-None-Match": '"sha256:other"'}
        )

        assert response.status_code == 200
        assert response.content == b"wav"
        assert [name for name, _ in catalog.calls] == ["find_derivative", "get_derivative"]

    def test_when_unknown_derivative_revalidated_then_404(self, client: TestClient) -> None:
        response = client.get("/derivatives/sha256:00", headers={"If-None-Match": "*"})

        assert response.status_code == 404


class TestGivenBadRequests:
    """Errors carry the one body shape and the status the cause deserves."""

    def test_when_unknown_resource_then_404(self, client: TestClient) -> None:
        for path in (
            "/scene-sets/sha256:00",
            "/scene-sets/sha256:00/scenes",
            f"/scene-sets/{SCENE_SET.version}/scenes/9",
            "/asset-versions/sha256:00",
            "/derivatives/sha256:00",
        ):
            response = client.get(path)
            assert response.status_code == 404, path
            assert response.json()["error"]["code"] == "not_found"
            assert response.headers["cache-control"] == "no-store"

    def test_when_unknown_path_then_404_same_shape(self, client: TestClient) -> None:
        response = client.get("/nothing/here")

        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found", "message": "Not Found"}}

    def test_when_method_not_allowed_then_405_with_allow(self, client: TestClient) -> None:
        response = client.post("/health")

        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"
        assert "GET" in response.headers["allow"]

    @pytest.mark.parametrize(
        "path, params",
        [
            ("/scene-sets", {"limit": "0"}),
            ("/scene-sets", {"limit": "many"}),
            ("/scene-sets", {"segmentation": "xyz"}),
            ("/scene-sets", {"enrichment": "transcript"}),
            (f"/scene-sets/{SCENE_SET.version}/scenes", {"after": "-1"}),
            (f"/scene-sets/{SCENE_SET.version}/scenes", {"limit": "0"}),
            (
                f"/scene-sets/{SCENE_SET.version}/scenes/0",
                {"enrichment": "Transcript:0123456789ab"},
            ),
            (f"/scene-sets/{SCENE_SET.version}/scenes/zero", {}),
        ],
    )
    def test_when_parameter_invalid_then_400(
        self, client: TestClient, path: str, params: dict[str, str]
    ) -> None:
        response = client.get(path, params=params)

        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "invalid_parameter"
        assert body["error"]["message"]

    def test_when_catalog_rejects_query_then_400(
        self, catalog: FakeSceneCatalog, client: TestClient
    ) -> None:
        catalog.failure = InvalidQueryError("limit must be at least 1")

        response = client.get("/scene-sets")

        assert response.status_code == 400
        assert response.json()["error"] == {
            "code": "invalid_parameter",
            "message": "limit must be at least 1",
        }

    def test_when_derivative_unavailable_then_503(
        self, catalog: FakeSceneCatalog, client: TestClient
    ) -> None:
        catalog.failure = DerivativeUnavailableError("file:///x: content does not match")

        response = client.get(f"/derivatives/{CLIP.digest}")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "derivative_unavailable"

    def test_when_catalog_raises_then_500_without_detail(
        self, catalog: FakeSceneCatalog, client: TestClient
    ) -> None:
        catalog.failure = RuntimeError("store went away: secret://user:pw@host")

        response = client.get("/scene-sets")

        assert response.status_code == 500
        assert response.json()["error"] == {
            "code": "internal_error",
            "message": "the request failed inside the service",
        }
        assert "secret://" not in response.text

    def test_when_server_raises_other_http_error_then_code_names_status(
        self, catalog: FakeSceneCatalog
    ) -> None:
        app = create_app(catalog)

        @app.get("/strict")
        def strict() -> None:
            raise HTTPException(400, "malformed")

        response = TestClient(app).get("/strict")

        assert response.status_code == 400
        assert response.json() == {"error": {"code": "bad_request", "message": "malformed"}}


class TestGivenEntityTags:
    """`If-None-Match` matching on its own."""

    @pytest.mark.parametrize(
        "header, expected",
        [
            (None, False),
            ('"a"', True),
            ('W/"a"', True),
            ('"b"', False),
            ('"b", "a"', True),
            ("*", True),
            ("", False),
        ],
    )
    def test_when_header_given_then_matched_as_rfc_9110_weakly_compares(
        self, header: str | None, expected: bool
    ) -> None:
        assert matches_etag(header, '"a"') is expected


class TestGivenPins:
    """Pin parsing on its own."""

    def test_when_none_then_empty(self) -> None:
        assert parse_pins(None) == ()

    def test_when_malformed_then_invalid_query(self) -> None:
        with pytest.raises(InvalidQueryError, match="enrichment pin"):
            parse_pins(["transcript:nope"])
