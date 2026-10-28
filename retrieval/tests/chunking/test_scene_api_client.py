"""The scene-api client over a scripted transport: paging, query strings, decoding, errors."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.http_transport import HttpStatusError
from scene_rag.retrieval.chunking.protocols import SceneApiError
from scene_rag.retrieval.chunking.scene_api_client import (
    SceneApiClient,
    asset_version_from_json,
    scene_record_from_json,
    summary_from_json,
)
from tests.support.chunking_fakes import (
    DESCRIPTION_FP,
    TRANSCRIPT_FP,
    encode_asset_version,
    encode_page,
    encode_scene_record,
    encode_summary,
    make_records,
    summary_of,
)

type Json = dict[str, Any]
type Answer = Json | Exception | Callable[[Mapping[str, list[str]]], Json]

BASE = "http://scene-api:8080"
RECORDS = make_records()
SUMMARY = summary_of(RECORDS)
SAMPLES = Path(__file__).resolve().parents[1] / "contract" / "samples"


class ScriptedTransport:
    """JsonTransport answering by path from a script; records every request."""

    def __init__(self, script: Mapping[str, Answer]) -> None:
        self.script = dict(script)
        self.requests: list[tuple[str, dict[str, list[str]], float]] = []

    def get(self, url: str, timeout: float) -> Json:
        parts = urlsplit(url)
        query: dict[str, list[str]] = {}
        for key, value in parse_qsl(parts.query):
            query.setdefault(key, []).append(value)
        self.requests.append((parts.path, query, timeout))
        answer = self.script.get(parts.path, HttpStatusError(404, url, "nothing here"))
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(query)
        return answer


def client(script: Mapping[str, Answer]) -> tuple[SceneApiClient, ScriptedTransport]:
    transport = ScriptedTransport(script)
    return SceneApiClient(BASE + "/", transport, 12.5), transport


class TestGivenListings:
    """Paged listings are walked to the end, with the query the step asked for."""

    def test_when_scene_sets_listed_then_pages_followed_and_query_encoded(self) -> None:
        first = encode_summary(SUMMARY)
        second = encode_summary(summary_of(make_records(bounds=((0.0, 1.0),))))

        def pages(query: Mapping[str, list[str]]) -> Json:
            if "after" not in query:
                return encode_page([first], "cursor-1")
            if query["after"] == ["cursor-1"]:
                return encode_page([], "cursor-2")
            return encode_page([second], None)

        api, transport = client({"/scene-sets": pages})
        query = models.SceneSetQuery(
            "3b9e1f6a2c40",
            (
                models.EnrichmentPin("description", DESCRIPTION_FP),
                models.EnrichmentPin("transcript", TRANSCRIPT_FP),
            ),
        )

        listed = list(api.list_scene_sets(query))

        assert [s.version for s in listed] == [first["version"], second["version"]]
        assert listed[0] == SUMMARY
        assert [(path, q.get("after"), t) for path, q, t in transport.requests] == [
            ("/scene-sets", None, 12.5),
            ("/scene-sets", ["cursor-1"], 12.5),
            ("/scene-sets", ["cursor-2"], 12.5),
        ]
        assert transport.requests[0][1] == {
            "segmentation": ["3b9e1f6a2c40"],
            "enrichment": [f"description:{DESCRIPTION_FP}", f"transcript:{TRANSCRIPT_FP}"],
        }

    def test_when_no_filters_then_bare_query(self) -> None:
        api, transport = client({"/scene-sets": encode_page([], None)})

        assert list(api.list_scene_sets(models.SceneSetQuery())) == []
        assert transport.requests == [("/scene-sets", {}, 12.5)]

    def test_when_records_listed_then_decoded_in_order_with_pins(self) -> None:
        path = f"/scene-sets/{SUMMARY.version}/scenes"
        encoded = [encode_scene_record(r) for r in RECORDS]

        def pages(query: Mapping[str, list[str]]) -> Json:
            if "after" not in query:
                return encode_page(encoded[:2], "1")
            return encode_page(encoded[2:], None)

        api, transport = client({path: pages})
        pins = [models.EnrichmentPin("transcript", TRANSCRIPT_FP)]

        records = api.list_scene_records(SUMMARY.version, pins)

        assert tuple(records) == RECORDS
        assert transport.requests[0][1] == {"enrichment": [f"transcript:{TRANSCRIPT_FP}"]}
        assert transport.requests[1][1]["after"] == ["1"]

    def test_when_scene_set_gone_then_error_propagates(self) -> None:
        api, _ = client({})

        with pytest.raises(HttpStatusError, match="404"):
            api.list_scene_records("sha256:00", [])

    @pytest.mark.parametrize(
        "page, message",
        [({"next": None}, "no items list"), ({"items": [], "next": 3}, "non-string cursor")],
    )
    def test_when_page_malformed_then_scene_api_error(self, page: Json, message: str) -> None:
        api, _ = client({"/scene-sets": page})

        with pytest.raises(SceneApiError, match=message):
            list(api.list_scene_sets(models.SceneSetQuery()))


class TestGivenAssetVersions:
    """Direct fetches of liveness."""

    def test_when_present_then_decoded(self) -> None:
        asset = models.AssetVersionView("sha256:ab", True)
        api, transport = client({"/asset-versions/sha256:ab": encode_asset_version(asset)})

        assert api.get_asset_version("sha256:ab") == asset
        assert transport.requests[0][0] == "/asset-versions/sha256:ab"

    def test_when_absent_then_none(self) -> None:
        api, _ = client({})

        assert api.get_asset_version("sha256:00") is None

    def test_when_server_fails_then_error_propagates(self) -> None:
        api, _ = client({"/asset-versions/sha256:ab": HttpStatusError(500, "u", "boom")})

        with pytest.raises(HttpStatusError, match="500"):
            api.get_asset_version("sha256:ab")


class TestGivenHealth:
    """The startup probe."""

    def test_when_ok_then_returns(self) -> None:
        api, transport = client({"/health": {"status": "ok"}})

        api.health()

        assert transport.requests == [("/health", {}, 5.0)]

    def test_when_not_ok_then_scene_api_error(self) -> None:
        api, _ = client({"/health": {"status": "draining"}})

        with pytest.raises(SceneApiError, match="draining"):
            api.health()


class TestGivenDecoders:
    """The published documents decode into the step's views."""

    def test_when_record_encoded_and_decoded_then_identical(self) -> None:
        for record in RECORDS:
            assert scene_record_from_json(encode_scene_record(record)) == record

    def test_when_summary_encoded_and_decoded_then_identical(self) -> None:
        assert summary_from_json(encode_summary(SUMMARY)) == SUMMARY

    def test_when_record_has_other_schema_version_then_refused(self) -> None:
        document = encode_scene_record(RECORDS[0])
        document["schema_version"] = 2

        with pytest.raises(SceneApiError, match="schema_version 2"):
            scene_record_from_json(document)

    def test_when_sample_asset_version_decoded_then_live(self) -> None:
        sample = json.loads((SAMPLES / "asset-version.json").read_text())

        assert asset_version_from_json(sample) == models.AssetVersionView(sample["version"], True)
