"""The step's read model over the scene set collection, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator

import pytest
from pymongo import MongoClient

from scene_rag.extraction.enrichment.models import SceneSetSummary
from scene_rag.extraction.enrichment.protocols import SceneSetReader
from scene_rag.extraction.enrichment.scene_store import (
    Document,
    MongoSceneSetReader,
    scene_set_from_document,
    summary_from_document,
)
from scene_rag.extraction.segmentation.scene_store import to_document
from tests.support.enrichment_fakes import FakeSceneSetReader
from tests.support.segmentation_fakes import make_scene_set, sha256_version

FIRST = make_scene_set(sha256_version(b"first"), "aaaaaaaaaaaa")
RECUT = make_scene_set(sha256_version(b"first"), "bbbbbbbbbbbb")


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def reader(request: pytest.FixtureRequest) -> Iterator[SceneSetReader]:
    documents = [to_document(FIRST), to_document(RECUT)]
    if request.param == "fake":
        yield FakeSceneSetReader(*(scene_set_from_document(d) for d in documents))
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    client[database]["scene_sets"].insert_many(documents)
    yield MongoSceneSetReader(client, database)
    client.drop_database(database)


class TestGivenStoredSceneSets:
    """Two sets of one asset version, cut under two fingerprints."""

    def test_when_summaries_listed_then_every_set_with_its_scene_count_in_version_order(
        self, reader: SceneSetReader
    ) -> None:
        assert list(reader.list_summaries()) == sorted(
            [
                SceneSetSummary(FIRST.version, FIRST.asset_version, 3),
                SceneSetSummary(RECUT.version, RECUT.asset_version, 3),
            ],
            key=lambda s: s.version,
        )

    def test_when_getting_then_scenes_with_derivatives(self, reader: SceneSetReader) -> None:
        view = reader.get(FIRST.version)

        assert view is not None
        assert view.asset_version == FIRST.asset_version
        assert [(s.index, s.start_seconds, s.end_seconds) for s in view.scenes] == [
            (0, 0.0, 4.0),
            (1, 4.0, 8.0),
            (2, 8.0, 12.5),
        ]
        assert (
            view.scenes[0].keyframes[0].derivative.digest
            == FIRST.scenes[0].keyframes[0].derivative.digest
        )
        assert view.scenes[0].keyframes[0].frame == 12
        assert view.scenes[0].audio_clip is not None
        assert view.scenes[0].audio_clip.media_type == "audio/wav"
        assert view.scenes[1].audio_clip is None
        assert reader.get("sha256:missing") is None


class TestGivenProjectedDocument:
    """The listing projection, as MongoDB returns it."""

    def test_when_decoded_then_summary(self) -> None:
        summary = summary_from_document(
            {"_id": "sha256:aa", "asset_version": "sha256:bb", "scene_count": 2}
        )

        assert summary == SceneSetSummary("sha256:aa", "sha256:bb", 2)
