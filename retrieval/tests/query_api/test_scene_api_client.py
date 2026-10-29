"""The scene-api client over a scripted transport: URLs, pass-through, bytes, errors."""

import json
from pathlib import Path
from typing import Any

import pytest

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.http_transport import HttpStatusError
from scene_rag.retrieval.query_api.protocols import SceneApiError
from scene_rag.retrieval.query_api.scene_api_client import SceneApiClient
from tests.support.query_fakes import FakeTransport

BASE = "http://scene-api:8080"
SAMPLES = Path(__file__).resolve().parents[1] / "contract" / "samples"
RECORD: dict[str, Any] = json.loads((SAMPLES / "scene-record.json").read_text())
SET = RECORD["scene_set_version"]
DIGEST = RECORD["scene"]["keyframes"][0]["digest"]


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport(
        {"GET /health": {"status": "ok"}, f"GET /scene-sets/{SET}/scenes/0": RECORD},
        {f"FETCH /derivatives/{DIGEST}": (b"\xff\xd8", "image/jpeg")},
        base=BASE,
    )


@pytest.fixture
def client(transport: FakeTransport) -> SceneApiClient:
    return SceneApiClient(BASE + "/", transport, timeout_seconds=12.5)


class TestGivenApi:
    """An API answering one scene record and one keyframe."""

    def test_when_health_checked_then_short_probe(
        self, client: SceneApiClient, transport: FakeTransport
    ) -> None:
        client.health()

        assert transport.calls == [("GET", BASE + "/health", None, 5.0)]

    def test_when_health_not_ok_then_error(
        self, client: SceneApiClient, transport: FakeTransport
    ) -> None:
        transport.answers["GET /health"] = {"status": "degraded"}

        with pytest.raises(SceneApiError, match="degraded"):
            client.health()

    def test_when_record_fetched_then_passed_through_unchanged(
        self, client: SceneApiClient, transport: FakeTransport
    ) -> None:
        record = client.get_scene_record(SET, 0)

        assert record == RECORD
        assert transport.calls[-1] == ("GET", f"{BASE}/scene-sets/{SET}/scenes/0", None, 12.5)

    def test_when_record_absent_then_none(self, client: SceneApiClient) -> None:
        assert client.get_scene_record(SET, 99) is None

    def test_when_record_schema_differs_then_error(
        self, client: SceneApiClient, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /scene-sets/{SET}/scenes/0"] = {**RECORD, "schema_version": 2}

        with pytest.raises(SceneApiError, match="schema_version 2"):
            client.get_scene_record(SET, 0)

    def test_when_derivative_fetched_then_bytes_with_media_type(
        self, client: SceneApiClient, transport: FakeTransport
    ) -> None:
        content = client.get_derivative(DIGEST)

        assert content == models.DerivativeContent("image/jpeg", b"\xff\xd8")
        assert transport.calls[-1] == ("FETCH", f"{BASE}/derivatives/{DIGEST}", None, 12.5)

    def test_when_derivative_absent_then_none(self, client: SceneApiClient) -> None:
        assert client.get_derivative("sha256:00") is None


class TestGivenFailingApi:
    """Error statuses and an API that is down."""

    def test_when_record_answers_other_status_then_error(
        self, client: SceneApiClient, transport: FakeTransport
    ) -> None:
        transport.answers[f"GET /scene-sets/{SET}/scenes/0"] = HttpStatusError(500, BASE, "boom")

        with pytest.raises(SceneApiError, match="HTTP 500"):
            client.get_scene_record(SET, 0)

    def test_when_derivative_answers_other_status_then_error(
        self, client: SceneApiClient, transport: FakeTransport
    ) -> None:
        transport.bodies[f"FETCH /derivatives/{DIGEST}"] = HttpStatusError(503, BASE, "later")

        with pytest.raises(SceneApiError, match="HTTP 503"):
            client.get_derivative(DIGEST)

    @pytest.mark.parametrize(
        "path", ["GET /health", f"GET /scene-sets/{SET}/scenes/0", f"FETCH /derivatives/{DIGEST}"]
    )
    def test_when_unreachable_then_error(
        self, client: SceneApiClient, transport: FakeTransport, path: str
    ) -> None:
        transport.unreachable(path)

        with pytest.raises(SceneApiError, match="could not reach"):
            if path == "GET /health":
                client.health()
            elif path.startswith("FETCH"):
                client.get_derivative(DIGEST)
            else:
                client.get_scene_record(SET, 0)
