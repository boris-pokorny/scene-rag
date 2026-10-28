"""Command semantics of the Scene Store, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator
from typing import Protocol

import pytest
from pymongo import MongoClient

from scene_rag.extraction.segmentation.protocols import SceneStoreReader, SceneStoreWriter
from scene_rag.extraction.segmentation.scene_store import (
    COLLECTION,
    Document,
    MongoSceneStore,
    from_document,
    to_document,
)
from tests.support.segmentation_fakes import T0, FakeSceneStore, make_scene_set, sha256_version


class SceneStore(SceneStoreReader, SceneStoreWriter, Protocol):
    """Both sides of the store, as every backend under test provides them."""


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def store(request: pytest.FixtureRequest) -> Iterator[SceneStore]:
    if request.param == "fake":
        yield FakeSceneStore()
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield MongoSceneStore(client, database)
    client.drop_database(database)


class TestGivenEmptyStore:
    """No scene set stored."""

    def test_when_put_then_get_returns_it(self, store: SceneStore) -> None:
        scene_set = make_scene_set()

        assert store.put(scene_set) is True
        assert store.get(scene_set.version) == scene_set
        assert store.get("sha256:missing") is None

    def test_when_nothing_stored_then_no_asset_versions(self, store: SceneStore) -> None:
        assert store.list_asset_versions("0123456789ab") == frozenset()


class TestGivenStoredSets:
    """Two sets under one fingerprint and one under another."""

    @pytest.fixture
    def stored(self, store: SceneStore) -> None:
        store.put(make_scene_set(sha256_version(b"a"), "aaaaaaaaaaaa"))
        store.put(make_scene_set(sha256_version(b"b"), "aaaaaaaaaaaa"))
        store.put(make_scene_set(sha256_version(b"a"), "bbbbbbbbbbbb"))

    def test_when_listed_by_fingerprint_then_its_asset_versions(
        self, store: SceneStore, stored: None
    ) -> None:
        assert store.list_asset_versions("aaaaaaaaaaaa") == {
            sha256_version(b"a"),
            sha256_version(b"b"),
        }
        assert store.list_asset_versions("bbbbbbbbbbbb") == {sha256_version(b"a")}

    def test_when_put_again_then_refused_and_unchanged(
        self, store: SceneStore, stored: None
    ) -> None:
        original = make_scene_set(sha256_version(b"a"), "aaaaaaaaaaaa")
        rival = make_scene_set(sha256_version(b"a"), "aaaaaaaaaaaa", scenes=())

        assert store.put(rival) is False
        assert store.get(original.version) == original


class TestGivenSceneSet:
    """A scene set with keyframes and one audio clip, encoded as a store document."""

    def test_when_encoded_and_decoded_then_equal(self) -> None:
        scene_set = make_scene_set()
        document = to_document(scene_set)

        assert document["_id"] == scene_set.version
        assert document["schema_version"] == 1
        assert document["lineage"]["step"] == "segmentation"
        assert document["scenes"][1]["audio_clip"] is None
        assert document["scenes"][0]["keyframes"][0]["frame"] == 12
        assert from_document(document) == scene_set

    def test_when_date_naive_then_utc_attached(self) -> None:
        scene_set = make_scene_set()
        document = to_document(scene_set)
        document["created_at"] = T0.replace(tzinfo=None)

        assert from_document(document) == scene_set


@pytest.mark.integration
class TestGivenMongoCollection:
    """The MongoDB collection as `MongoSceneStore` prepares it."""

    def test_when_created_then_fingerprint_lookup_indexed(
        self, mongo_client: MongoClient[Document]
    ) -> None:
        database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
        try:
            MongoSceneStore(mongo_client, database)
            indexes = mongo_client[database][COLLECTION].index_information().values()
            keys = [tuple(index["key"]) for index in indexes]
        finally:
            mongo_client.drop_database(database)

        assert (("lineage.fingerprint", 1), ("asset_version", 1)) in keys
        assert (("asset_version", 1),) in keys
        assert (("scenes.keyframes.derivative.digest", 1),) in keys
        assert (("scenes.audio_clip.digest", 1),) in keys
