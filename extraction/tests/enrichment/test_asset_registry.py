"""The step's read model over the registry collection, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator
from datetime import UTC

import pytest
from pymongo import MongoClient

from scene_rag.extraction.enrichment.asset_registry import (
    Document,
    MongoAssetVersionReader,
    view_from_document,
)
from scene_rag.extraction.enrichment.protocols import AssetVersionReader
from scene_rag.extraction.ingestion.registry import to_document
from tests.support.enrichment_fakes import FakeAssetVersionReader
from tests.support.fakes import T0, make_record, make_reference, sha256_version

LIVE = make_record(sha256_version(b"live"), make_reference("asset-0000000000000001"))
RETIRED = make_record(sha256_version(b"gone"), retired_at=T0)
# Written by a crashed pass: every reference removed, tombstone not yet set.
ORPHAN = make_record(sha256_version(b"orphan"))


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def reader(request: pytest.FixtureRequest) -> Iterator[AssetVersionReader]:
    documents = [to_document(LIVE), to_document(RETIRED), {**to_document(ORPHAN), "sources": []}]
    if request.param == "fake":
        yield FakeAssetVersionReader(*(view_from_document(d) for d in documents))
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    client[database]["asset_versions"].insert_many(documents)
    yield MongoAssetVersionReader(client, database)
    client.drop_database(database)


class TestGivenRegistryWithLiveAndRetiredVersions:
    """Records as ingestion wrote them."""

    def test_when_listing_live_then_only_held_unretired_versions(
        self, reader: AssetVersionReader
    ) -> None:
        assert reader.list_live_versions() == {LIVE.version}

    def test_when_getting_then_live_or_retired_view(self, reader: AssetVersionReader) -> None:
        live = reader.get(LIVE.version)
        retired = reader.get(RETIRED.version)
        orphan = reader.get(ORPHAN.version)

        assert live is not None and live.is_live and live.retired_at is None
        assert retired is not None and not retired.is_live and retired.retired_at == T0
        assert orphan is not None and not orphan.is_live and orphan.retired_at is None
        assert reader.get("sha256:missing") is None


class TestGivenRegistryDocument:
    """A document straight from pymongo, with naive datetimes."""

    def test_when_decoded_then_only_needed_fields_and_utc_attached(self) -> None:
        document = to_document(RETIRED)
        document["retired_at"] = document["retired_at"].replace(tzinfo=None)

        view = view_from_document(document)

        assert view.version == RETIRED.version
        assert view.has_sources is True and not view.is_live
        assert view.retired_at is not None and view.retired_at.tzinfo is UTC
