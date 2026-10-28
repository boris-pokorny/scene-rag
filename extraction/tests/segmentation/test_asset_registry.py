"""The step's read model over the registry collection, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator
from datetime import timedelta

import pytest
from pymongo import MongoClient

from scene_rag.extraction.ingestion.registry import to_document
from scene_rag.extraction.segmentation.asset_registry import (
    Document,
    MongoAssetVersionReader,
    view_from_document,
)
from scene_rag.extraction.segmentation.protocols import AssetVersionReader
from tests.support.fakes import T0, make_record, make_reference, sha256_version
from tests.support.segmentation_fakes import FakeAssetVersionReader, make_source

LIVE = make_record(sha256_version(b"live"), make_reference("asset-0000000000000001"))
SHARED = make_record(
    sha256_version(b"shared"),
    make_reference("asset-0000000000000002"),
    make_reference("asset-0000000000000003", source_id="library"),
)
RETIRED = make_record(sha256_version(b"gone"), retired_at=T0)
# Written by a crashed pass: every reference removed, tombstone not yet set.
ORPHAN = make_record(sha256_version(b"orphan"))


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def reader(request: pytest.FixtureRequest) -> Iterator[AssetVersionReader]:
    if request.param == "fake":
        yield FakeAssetVersionReader(
            view_from_document(to_document(LIVE)),
            view_from_document(to_document(SHARED)),
            view_from_document(to_document(RETIRED)),
            view_from_document({**to_document(ORPHAN), "sources": []}),
        )
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    collection = client[database]["asset_versions"]
    collection.insert_many(
        [
            to_document(LIVE),
            to_document(SHARED),
            to_document(RETIRED),
            {**to_document(ORPHAN), "sources": []},
        ]
    )
    yield MongoAssetVersionReader(client, database)
    client.drop_database(database)


class TestGivenRegistryWithLiveAndRetiredVersions:
    """Records as ingestion wrote them."""

    def test_when_listing_live_then_only_held_unretired_versions_in_order(
        self, reader: AssetVersionReader
    ) -> None:
        live = reader.list_live()

        assert [v.version for v in live] == sorted([LIVE.version, SHARED.version])
        assert {v.version: len(v.sources) for v in live} == {LIVE.version: 1, SHARED.version: 2}

    def test_when_getting_then_live_or_retired_view(self, reader: AssetVersionReader) -> None:
        shared = reader.get(SHARED.version)
        retired = reader.get(RETIRED.version)

        assert shared is not None and shared.is_live
        assert [(s.source_id, s.asset_id) for s in shared.sources] == [
            ("local", "asset-0000000000000002"),
            ("library", "asset-0000000000000003"),
        ]
        assert shared.sources[0] == make_source(
            "asset-0000000000000002", "file:///media/asset-0000000000000002.mp4", 4, T0
        )
        assert retired is not None and retired.retired_at == T0 and not retired.is_live
        assert reader.get("sha256:missing") is None


class TestGivenRegistryDocument:
    """One registry document, decoded."""

    def test_when_decoded_then_only_needed_fields_and_utc_attached(self) -> None:
        document = to_document(LIVE)
        document["sources"][0]["modified_at"] = (T0 + timedelta(hours=1)).replace(tzinfo=None)

        view = view_from_document(document)

        assert view.version == LIVE.version
        assert view.media_type == "video/mp4" and view.size_bytes == 4
        (source,) = view.sources
        assert source.modified_at == T0 + timedelta(hours=1)
        assert source.modified_at.tzinfo is not None
        assert view.retired_at is None
