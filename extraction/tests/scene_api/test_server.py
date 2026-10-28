"""End to end over MongoDB: the stores as the steps write them, served by the real wiring."""

import hashlib
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient

from scene_rag.extraction.enrichment.enrichment_store import to_document as enrichment_document
from scene_rag.extraction.enrichment.scene_store import scene_set_from_document
from scene_rag.extraction.ingestion.registry import to_document as asset_document
from scene_rag.extraction.scene_api import bootstrap
from scene_rag.extraction.scene_api.app import create_app
from scene_rag.extraction.scene_api.asset_registry import Document
from scene_rag.extraction.scene_api.settings import SceneApiSettings
from scene_rag.extraction.segmentation import models as segmentation
from scene_rag.extraction.segmentation.scene_store import to_document as scene_set_document
from tests.support.enrichment_fakes import make_enrichment
from tests.support.fakes import T0, make_record, make_reference
from tests.support.segmentation_fakes import make_scene_set, make_segments, sha256_version

pytestmark = pytest.mark.integration

LIVE_ASSET = make_record(
    sha256_version(b"live"), make_reference("asset-0000000000000001", source_id="local")
)
RETIRED_ASSET = make_record(sha256_version(b"gone"), retired_at=T0 + timedelta(hours=1))
TRANSCRIPT_FP = "e41a9c27b0d5"


def filed(directory: Path, data: bytes, media_type: str) -> segmentation.DerivativeRef:
    digest = hashlib.sha256(data).hexdigest()
    path = directory / f"{digest}.{'jpg' if media_type == 'image/jpeg' else 'wav'}"
    path.write_bytes(data)
    return segmentation.DerivativeRef(
        digest=f"sha256:{digest}", uri=path.as_uri(), media_type=media_type, size_bytes=len(data)
    )


def scene_set_with_files(asset_version: str, directory: Path) -> segmentation.SceneSet:
    scenes = tuple(
        segmentation.Scene(
            index=segment.index,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            keyframes=(
                segmentation.Keyframe(
                    derivative=filed(directory, f"jpeg {segment.index}".encode(), "image/jpeg"),
                    frame=segment.start_frame + 12,
                    seconds=segment.start_seconds + 0.48,
                ),
            ),
            audio_clip=filed(directory, f"wav {segment.index}".encode(), "audio/wav"),
        )
        for segment in make_segments()
    )
    return make_scene_set(asset_version, "0123456789ab", scenes=scenes)


@pytest.fixture(scope="module")
def derivatives(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("derivatives")


@pytest.fixture(scope="module")
def live_set(derivatives: Path) -> segmentation.SceneSet:
    return scene_set_with_files(LIVE_ASSET.version, derivatives)


@pytest.fixture(scope="module")
def retired_set(derivatives: Path) -> segmentation.SceneSet:
    return scene_set_with_files(RETIRED_ASSET.version, derivatives)


@pytest.fixture(scope="module")
def client(
    mongo_client: MongoClient[Document],
    mongo_uri: str,
    live_set: segmentation.SceneSet,
    retired_set: segmentation.SceneSet,
) -> Iterator[TestClient]:
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    db = mongo_client[database]
    db["asset_versions"].insert_many(
        [asset_document(LIVE_ASSET), {**asset_document(RETIRED_ASSET), "sources": []}]
    )
    db["scene_sets"].insert_many([scene_set_document(live_set), scene_set_document(retired_set)])
    view = scene_set_from_document(scene_set_document(live_set))
    db["enrichments"].insert_many(
        [
            enrichment_document(make_enrichment(view, index, "transcript", TRANSCRIPT_FP))
            for index in range(3)
        ]
        + [enrichment_document(make_enrichment(view, 0, "description", "3b9e1f6a2c40"))]
    )
    settings = SceneApiSettings(
        registry_uri=mongo_uri,
        registry_database=database,
        scene_store_uri=mongo_uri,
        scene_store_database=database,
        enrichment_store_uri=mongo_uri,
        enrichment_store_database=database,
        page_size=2,
    )
    catalog = bootstrap.create_catalog(settings, mongo_client, mongo_client, mongo_client)
    yield TestClient(create_app(catalog), raise_server_exceptions=False)
    mongo_client.drop_database(database)


class TestGivenPopulatedStores:
    """A live and a retired asset version, each cut once; the live one enriched."""

    def test_when_health_then_ok(self, client: TestClient) -> None:
        assert client.get("/health").json() == {"status": "ok"}

    def test_when_sets_listed_then_only_live_with_coverage(
        self, client: TestClient, live_set: segmentation.SceneSet
    ) -> None:
        body = client.get("/scene-sets").json()

        assert [item["version"] for item in body["items"]] == [live_set.version]
        assert body["items"][0]["enrichments"] == [
            {
                "enricher": "description",
                "fingerprint": "3b9e1f6a2c40",
                "scene_count": 1,
                "complete": False,
            },
            {
                "enricher": "transcript",
                "fingerprint": TRANSCRIPT_FP,
                "scene_count": 3,
                "complete": True,
            },
        ]

    def test_when_pinned_then_completeness_decides(
        self, client: TestClient, live_set: segmentation.SceneSet
    ) -> None:
        complete = client.get("/scene-sets", params={"enrichment": f"transcript:{TRANSCRIPT_FP}"})
        partial = client.get("/scene-sets", params={"enrichment": "description:3b9e1f6a2c40"})

        assert [i["version"] for i in complete.json()["items"]] == [live_set.version]
        assert partial.json()["items"] == []

    def test_when_records_walked_then_every_scene_once_with_pinned_enrichment(
        self, client: TestClient, live_set: segmentation.SceneSet
    ) -> None:
        indices: list[int] = []
        cursor: str | None = None
        while True:
            params = {"enrichment": f"transcript:{TRANSCRIPT_FP}"}
            if cursor is not None:
                params["after"] = cursor
            body = client.get(f"/scene-sets/{live_set.version}/scenes", params=params).json()
            for record in body["items"]:
                indices.append(record["scene"]["index"])
                assert [e["enricher"] for e in record["enrichments"]] == ["transcript"]
                assert record["asset"]["entitlements"] == ["public"]
                assert record["asset"]["sources"][0]["asset_id"] == "asset-0000000000000001"
                assert record["live"] is True
            if body["next"] is None:
                break
            cursor = body["next"]

        assert indices == [0, 1, 2]

    def test_when_retired_set_fetched_then_not_live(
        self, client: TestClient, retired_set: segmentation.SceneSet
    ) -> None:
        summary = client.get(f"/scene-sets/{retired_set.version}").json()
        record = client.get(f"/scene-sets/{retired_set.version}/scenes/0").json()
        asset = client.get(f"/asset-versions/{RETIRED_ASSET.version}").json()

        assert summary["live"] is False
        assert record["live"] is False and record["asset"]["sources"] == []
        assert asset["live"] is False and asset["retired_at"] == "2026-10-27T10:00:00Z"

    def test_when_derivative_fetched_then_file_bytes_served(
        self, client: TestClient, live_set: segmentation.SceneSet
    ) -> None:
        record = client.get(f"/scene-sets/{live_set.version}/scenes/1").json()
        keyframe = record["scene"]["keyframes"][0]

        response = client.get(keyframe["url"])

        assert response.status_code == 200
        assert response.content == b"jpeg 1"
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["etag"] == f'"{keyframe["digest"]}"'

    def test_when_file_tampered_then_503(
        self, client: TestClient, live_set: segmentation.SceneSet, derivatives: Path
    ) -> None:
        clip = live_set.scenes[2].audio_clip
        assert clip is not None
        path = derivatives / f"{clip.digest.split(':')[1]}.wav"
        path.write_bytes(b"other")
        try:
            response = client.get(f"/derivatives/{clip.digest}")
        finally:
            path.write_bytes(b"wav 2")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "derivative_unavailable"
