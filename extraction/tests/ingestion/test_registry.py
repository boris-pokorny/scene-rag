"""Command semantics of the Asset Registry, on the fake and on MongoDB.

The MongoDB case carries the `integration` marker and is skipped when no server answers.
"""

import uuid
from collections.abc import Iterator
from datetime import timedelta
from typing import Protocol

import pytest
from pymongo import MongoClient

from scene_rag.extraction.ingestion import models
from scene_rag.extraction.ingestion.protocols import (
    AssetRegistryReader,
    AssetRegistryWriter,
    UnknownVersionError,
)
from scene_rag.extraction.ingestion.registry import (
    COLLECTION,
    Document,
    MongoAssetRegistry,
    from_document,
    to_document,
)
from tests.support.fakes import T0, make_record, make_reference, sha256_version

LATER = T0 + timedelta(minutes=5)
LOCAL = make_reference("asset-0000000000000001")
LIBRARY = make_reference("asset-0000000000000002", source_id="library")


class Registry(AssetRegistryReader, AssetRegistryWriter, Protocol):
    """Both sides of the registry, as every backend under test provides them."""


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def registry(request: pytest.FixtureRequest) -> Iterator[Registry]:
    if request.param == "fake":
        yield request.getfixturevalue("fake_registry")
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield MongoAssetRegistry(client, database)
    client.drop_database(database)


class TestGivenEmptyRegistry:
    """No version stored."""

    def test_when_version_registered_then_get_returns_it(self, registry: Registry) -> None:
        record = make_record()

        assert registry.register_version(record) is True
        assert registry.get(record.version) == record
        assert registry.get("sha256:missing") is None

    def test_when_unknown_version_commanded_then_raises(self, registry: Registry) -> None:
        with pytest.raises(UnknownVersionError):
            registry.add_source_reference("sha256:missing", make_reference())
        with pytest.raises(UnknownVersionError):
            registry.remove_source_reference("sha256:missing", "local", "asset-x", at=T0)


class TestGivenStoredVersion:
    """One live version held by `LOCAL` and `LIBRARY`."""

    @pytest.fixture
    def stored(self, registry: Registry) -> models.AssetVersion:
        record = make_record(sha256_version(b"x"), LOCAL, LIBRARY)
        registry.register_version(record)
        return record

    def test_when_registered_again_then_refused_and_unchanged(
        self, registry: Registry, stored: models.AssetVersion
    ) -> None:
        rival = make_record(stored.version, make_reference("asset-0000000000000009"))

        assert registry.register_version(rival) is False
        assert registry.get(stored.version) == stored

    def test_when_all_references_removed_then_tombstoned_after_last(
        self, registry: Registry, stored: models.AssetVersion
    ) -> None:
        after_first = registry.remove_source_reference(stored.version, *LOCAL.key, at=T0)
        assert after_first.sources == (LIBRARY,)
        assert after_first.retired_at is None

        after_second = registry.remove_source_reference(stored.version, *LIBRARY.key, at=LATER)
        assert after_second.sources == ()
        assert after_second.retired_at == LATER
        assert not after_second.is_live
        assert registry.list_by_source("local") == []

    def test_when_listed_by_source_then_live_ones_of_source(
        self, registry: Registry, stored: models.AssetVersion
    ) -> None:
        registry.register_version(make_record(sha256_version(b"gone"), LOCAL, retired_at=T0))

        assert registry.list_by_source("local") == [stored]
        assert registry.list_by_source("library") == [stored]
        assert registry.list_by_source("nobody") == []


class TestGivenRetiredVersion:
    """A version whose only reference, `LOCAL`, was removed at `T0`."""

    @pytest.fixture
    def retired(self, registry: Registry) -> models.AssetVersion:
        record = make_record(sha256_version(b"x"), LOCAL)
        registry.register_version(record)
        return registry.remove_source_reference(record.version, *LOCAL.key, at=T0)

    def test_when_reference_added_then_replaces_key_and_unretires(
        self, registry: Registry, retired: models.AssetVersion
    ) -> None:
        refreshed = make_reference(LOCAL.asset_id, seen_at=LATER)

        registry.add_source_reference(retired.version, refreshed)
        registry.add_source_reference(retired.version, LIBRARY)

        updated = registry.get(retired.version)
        assert updated is not None
        assert updated.sources == (refreshed, LIBRARY)
        assert updated.retired_at is None
        assert updated.is_live

    def test_when_removed_again_then_tombstone_kept(
        self, registry: Registry, retired: models.AssetVersion
    ) -> None:
        again = registry.remove_source_reference(retired.version, *LOCAL.key, at=LATER)

        assert again.retired_at == T0


class TestGivenRecord:
    """A tombstoned record with two references, encoded as a registry document."""

    @pytest.fixture
    def record(self) -> models.AssetVersion:
        return make_record(
            sha256_version(b"x"),
            make_reference("asset-0000000000000001", entitlements=frozenset({"b", "a"})),
            LIBRARY,
            retired_at=T0,
        )

    def test_when_encoded_and_decoded_then_equal(self, record: models.AssetVersion) -> None:
        document = to_document(record)

        assert document["_id"] == record.version
        assert document["schema_version"] == 1
        assert document["sources"][0]["entitlements"] == ["a", "b"]
        assert document["lineage"]["step"] == "ingestion"
        assert from_document(document) == record

    def test_when_dates_naive_then_utc_attached(self, record: models.AssetVersion) -> None:
        document = to_document(record)
        document["first_seen_at"] = T0.replace(tzinfo=None)
        document["retired_at"] = T0.replace(tzinfo=None)
        for source in document["sources"]:
            source["seen_at"] = T0.replace(tzinfo=None)
            source["modified_at"] = T0.replace(tzinfo=None)

        assert from_document(document) == record


@pytest.mark.integration
class TestGivenMongoCollection:
    """The MongoDB collection as `MongoAssetRegistry` prepares it."""

    def test_when_created_then_list_by_source_served_by_compound_index(
        self, mongo_client: MongoClient[Document]
    ) -> None:
        database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
        try:
            MongoAssetRegistry(mongo_client, database)
            indexes = mongo_client[database][COLLECTION].index_information().values()
            keys = [tuple(index["key"]) for index in indexes]
        finally:
            mongo_client.drop_database(database)

        assert (("retired_at", 1), ("sources.source_id", 1)) in keys
        assert (("retired_at", 1),) not in keys
