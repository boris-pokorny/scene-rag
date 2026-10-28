"""The role's read model over the registry collection, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator
from datetime import timedelta

import pytest
from pymongo import MongoClient

from scene_rag.extraction.ingestion.registry import to_document
from scene_rag.extraction.scene_api.asset_registry import (
    Document,
    MongoAssetVersionReader,
    view_from_document,
)
from scene_rag.extraction.scene_api.models import AssetVersionView, SourceView
from scene_rag.extraction.scene_api.protocols import AssetVersionReader
from tests.support.fakes import T0, make_record, make_reference, sha256_version
from tests.support.scene_api_fakes import FakeAssetVersionReader

LIVE = make_record(
    sha256_version(b"live"),
    make_reference("asset-0000000000000001", entitlements=frozenset({"public"})),
    make_reference(
        "asset-0000000000000002",
        source_id="library",
        entitlements=frozenset({"team-platform", "public"}),
    ),
)
RETIRED = make_record(sha256_version(b"gone"), retired_at=T0 + timedelta(hours=1))
RETIRED_DOCUMENT = {**to_document(RETIRED), "sources": []}


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def reader(request: pytest.FixtureRequest) -> Iterator[AssetVersionReader]:
    documents = [to_document(LIVE), RETIRED_DOCUMENT]
    if request.param == "fake":
        yield FakeAssetVersionReader(*(view_from_document(d) for d in documents))
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    client[database]["asset_versions"].insert_many(documents)
    yield MongoAssetVersionReader(client, database)
    client.drop_database(database)


class TestGivenRegistryDocuments:
    """A live version held by two sources, and a retired one."""

    def test_when_live_fetched_then_ids_and_union_of_entitlements(
        self, reader: AssetVersionReader
    ) -> None:
        view = reader.get(LIVE.version)

        assert view == AssetVersionView(
            version=LIVE.version,
            media_type="video/mp4",
            size_bytes=LIVE.size_bytes,
            sources=(
                SourceView("local", "asset-0000000000000001", LIVE.sources[0].filename),
                SourceView("library", "asset-0000000000000002", LIVE.sources[1].filename),
            ),
            entitlements=frozenset({"public", "team-platform"}),
            retired_at=None,
        )
        assert view is not None and view.is_live

    def test_when_retired_fetched_then_tombstone_and_nothing_held(
        self, reader: AssetVersionReader
    ) -> None:
        view = reader.get(RETIRED.version)

        assert view is not None
        assert view.retired_at == T0 + timedelta(hours=1)
        assert view.sources == () and view.entitlements == frozenset()
        assert not view.is_live

    def test_when_unknown_fetched_then_none(self, reader: AssetVersionReader) -> None:
        assert reader.get("sha256:0000") is None

    def test_when_many_fetched_then_found_ones_keyed(self, reader: AssetVersionReader) -> None:
        found = reader.get_many([LIVE.version, "sha256:0000", RETIRED.version])

        assert set(found) == {LIVE.version, RETIRED.version}
        assert found[LIVE.version] == reader.get(LIVE.version)

    def test_when_none_asked_then_empty(self, reader: AssetVersionReader) -> None:
        assert reader.get_many([]) == {}
