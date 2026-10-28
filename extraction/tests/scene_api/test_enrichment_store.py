"""The role's read model over the enrichment collection, on the fake and on MongoDB."""

import uuid
from collections.abc import Iterator

import pytest
from pymongo import MongoClient

from scene_rag.extraction.enrichment.enrichment_store import to_document
from scene_rag.extraction.enrichment.models import SemanticSegment
from scene_rag.extraction.scene_api.enrichment_store import (
    Document,
    MongoEnrichmentReader,
    enrichment_from_document,
)
from scene_rag.extraction.scene_api.models import (
    Coverage,
    EnrichmentPin,
    ModelRef,
    SemanticSegmentView,
)
from scene_rag.extraction.scene_api.protocols import EnrichmentReader
from tests.support.enrichment_fakes import make_enrichment, make_scene_set, sha256_version
from tests.support.scene_api_fakes import FakeEnrichmentReader

FIRST = make_scene_set(sha256_version(b"first"))
SECOND = make_scene_set(sha256_version(b"second"))
SEGMENTS = (SemanticSegment(0.5, 1.5, "hello", "en", 0.9), SemanticSegment(1.5, 4.0, "there"))
RECORDS = [
    make_enrichment(FIRST, 0, "transcript", segments=SEGMENTS),
    make_enrichment(FIRST, 1, "transcript"),
    make_enrichment(FIRST, 2, "transcript"),
    make_enrichment(FIRST, 0, "description"),
    make_enrichment(FIRST, 0, "description", fingerprint="ffffffffffff"),
    make_enrichment(SECOND, 1, "transcript"),
]


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def reader(request: pytest.FixtureRequest) -> Iterator[EnrichmentReader]:
    documents = [to_document(record) for record in RECORDS]
    if request.param == "fake":
        yield FakeEnrichmentReader(*(enrichment_from_document(d) for d in documents))
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    client[database]["enrichments"].insert_many(documents)
    yield MongoEnrichmentReader(client, database)
    client.drop_database(database)


class TestGivenStoredRecords:
    """FIRST transcribed in full and described twice on scene 0; SECOND transcribed on scene 1."""

    def test_when_coverage_listed_then_distinct_scenes_per_enricher_and_fingerprint(
        self, reader: EnrichmentReader
    ) -> None:
        transcript_fp = RECORDS[0].lineage.fingerprint
        description_fp = RECORDS[3].lineage.fingerprint

        coverage = reader.list_coverage([FIRST.version, SECOND.version, "sha256:0000"])

        assert sorted(
            coverage, key=lambda c: (c.scene_set_version, c.enricher, c.fingerprint)
        ) == sorted(
            [
                Coverage(FIRST.version, "description", description_fp, 1),
                Coverage(FIRST.version, "description", "ffffffffffff", 1),
                Coverage(FIRST.version, "transcript", transcript_fp, 3),
                Coverage(SECOND.version, "transcript", transcript_fp, 1),
            ],
            key=lambda c: (c.scene_set_version, c.enricher, c.fingerprint),
        )

    def test_when_no_sets_asked_then_no_coverage(self, reader: EnrichmentReader) -> None:
        assert list(reader.list_coverage([])) == []

    def test_when_scenes_listed_then_ordered_and_decoded(self, reader: EnrichmentReader) -> None:
        records = reader.list_for_scenes(FIRST.version, [0, 2, 7])

        assert [(r.scene_index, r.enricher, r.fingerprint) for r in records] == [
            (0, "description", RECORDS[3].lineage.fingerprint),
            (0, "description", "ffffffffffff"),
            (0, "transcript", RECORDS[0].lineage.fingerprint),
            (2, "transcript", RECORDS[2].lineage.fingerprint),
        ]
        transcript = records[2]
        assert transcript.version == RECORDS[0].version
        assert transcript.classification == RECORDS[0].classification
        assert transcript.model == ModelRef(RECORDS[0].model.name, RECORDS[0].model.digest)
        assert transcript.segments == (
            SemanticSegmentView(0.5, 1.5, "hello", "en", 0.9),
            SemanticSegmentView(1.5, 4.0, "there", None, None),
        )

    def test_when_pinned_then_only_records_under_the_pins(self, reader: EnrichmentReader) -> None:
        pins = [
            EnrichmentPin("description", "ffffffffffff"),
            EnrichmentPin("transcript", RECORDS[0].lineage.fingerprint),
        ]

        records = reader.list_for_scenes(FIRST.version, [0, 1, 2], pins)

        assert [(r.scene_index, r.enricher, r.fingerprint) for r in records] == [
            (0, "description", "ffffffffffff"),
            (0, "transcript", RECORDS[0].lineage.fingerprint),
            (1, "transcript", RECORDS[1].lineage.fingerprint),
            (2, "transcript", RECORDS[2].lineage.fingerprint),
        ]

    def test_when_pinned_to_unknown_fingerprint_then_nothing(
        self, reader: EnrichmentReader
    ) -> None:
        pins = [EnrichmentPin("transcript", "000000000000")]

        assert list(reader.list_for_scenes(FIRST.version, [0, 1, 2], pins)) == []

    def test_when_other_set_asked_then_its_records_only(self, reader: EnrichmentReader) -> None:
        records = reader.list_for_scenes(SECOND.version, [0, 1, 2])

        assert [r.scene_index for r in records] == [1]
