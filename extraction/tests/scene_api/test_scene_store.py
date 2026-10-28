"""The role's read model over the scene set collection, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator
from datetime import timedelta

import pytest
from pymongo import MongoClient

from scene_rag.extraction.scene_api.models import SceneSetSummaryView
from scene_rag.extraction.scene_api.protocols import SceneSetReader
from scene_rag.extraction.scene_api.scene_store import (
    Document,
    MongoSceneSetReader,
    scene_set_from_document,
)
from scene_rag.extraction.segmentation.scene_store import to_document
from tests.support.scene_api_fakes import FakeSceneSetReader
from tests.support.segmentation_fakes import T0, make_scene_set, sha256_version

FIRST = make_scene_set(sha256_version(b"first"), "aaaaaaaaaaaa")
RECUT = make_scene_set(sha256_version(b"first"), "bbbbbbbbbbbb", created_at=T0 + timedelta(1))
SECOND = make_scene_set(sha256_version(b"second"), "aaaaaaaaaaaa")
ALL = sorted((FIRST, RECUT, SECOND), key=lambda s: s.version)


def summary_of(scene_set: object) -> SceneSetSummaryView:
    return scene_set_from_document(to_document(scene_set)).summary  # type: ignore[arg-type]


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def reader(request: pytest.FixtureRequest) -> Iterator[SceneSetReader]:
    documents = [to_document(s) for s in (FIRST, RECUT, SECOND)]
    if request.param == "fake":
        yield FakeSceneSetReader(*(scene_set_from_document(d) for d in documents))
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    client[database]["scene_sets"].insert_many(documents)
    yield MongoSceneSetReader(client, database)
    client.drop_database(database)


class TestGivenStoredSceneSets:
    """Three sets: one asset version cut twice, another once."""

    def test_when_fetched_then_decoded_with_media_frames_and_fingerprint(
        self, reader: SceneSetReader
    ) -> None:
        view = reader.get(FIRST.version)

        assert view is not None
        assert view.asset_version == FIRST.asset_version
        assert view.segmentation_fingerprint == "aaaaaaaaaaaa"
        assert view.created_at == T0
        assert (view.media.width, view.media.height, view.media.frame_count) == (1920, 1080, 312)
        assert [(s.index, s.start_frame, s.end_frame) for s in view.scenes] == [
            (0, 0, 100),
            (1, 100, 200),
            (2, 200, 312),
        ]
        assert view.scenes[0].audio_clip is not None and view.scenes[1].audio_clip is None
        assert (
            view.scenes[0].keyframes[0].derivative.digest
            == FIRST.scenes[0].keyframes[0].derivative.digest
        )

    def test_when_unknown_fetched_then_none(self, reader: SceneSetReader) -> None:
        assert reader.get("sha256:0000") is None

    def test_when_listed_then_summaries_in_version_order(self, reader: SceneSetReader) -> None:
        summaries = reader.list_summaries(
            after=None, limit=10, asset_version=None, segmentation_fingerprint=None
        )

        assert list(summaries) == [summary_of(s) for s in ALL]
        assert summaries[0].scene_count == 3

    def test_when_listed_after_cursor_with_limit_then_window(self, reader: SceneSetReader) -> None:
        summaries = reader.list_summaries(
            after=ALL[0].version, limit=1, asset_version=None, segmentation_fingerprint=None
        )

        assert [s.version for s in summaries] == [ALL[1].version]

    def test_when_filtered_then_only_matching(self, reader: SceneSetReader) -> None:
        by_asset = reader.list_summaries(
            after=None, limit=10, asset_version=FIRST.asset_version, segmentation_fingerprint=None
        )
        by_fingerprint = reader.list_summaries(
            after=None, limit=10, asset_version=None, segmentation_fingerprint="bbbbbbbbbbbb"
        )
        both = reader.list_summaries(
            after=None,
            limit=10,
            asset_version=SECOND.asset_version,
            segmentation_fingerprint="bbbbbbbbbbbb",
        )

        assert {s.version for s in by_asset} == {FIRST.version, RECUT.version}
        assert [s.version for s in by_fingerprint] == [RECUT.version]
        assert list(both) == []

    def test_when_derivative_looked_up_then_reference_found(self, reader: SceneSetReader) -> None:
        keyframe = FIRST.scenes[0].keyframes[0].derivative
        clip = FIRST.scenes[0].audio_clip
        assert clip is not None

        found_keyframe = reader.find_derivative(keyframe.digest)
        found_clip = reader.find_derivative(clip.digest)

        assert found_keyframe is not None and found_keyframe.digest == keyframe.digest
        assert found_keyframe.media_type == "image/jpeg"
        assert found_clip is not None and found_clip.uri == clip.uri
        assert found_clip.media_type == "audio/wav"
        assert reader.find_derivative("sha256:0000") is None
