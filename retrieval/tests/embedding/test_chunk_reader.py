"""The read model over the Chunk Store, on the fake and on MongoDB fed by the chunking step."""

import uuid
from collections.abc import Iterator

import pytest
from pymongo import MongoClient

from scene_rag.retrieval.chunking import models as chunking_models
from scene_rag.retrieval.chunking.chunk_store import (
    MongoChunkStore,
    chunk_set_to_document,
    chunk_to_document,
)
from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.chunk_reader import (
    CHUNK_SETS_COLLECTION,
    CHUNKS_COLLECTION,
    Document,
    MongoChunkReader,
)
from scene_rag.retrieval.embedding.protocols import ChunkReader
from tests.support import chunking_fakes
from tests.support.embedding_fakes import FakeChunkReader

# The chunks the chunking step would write for two scene sets under two fingerprints.
FIRST = chunking_fakes.make_records(chunking_fakes.sha256_version(b"first"))
SECOND = chunking_fakes.make_records(chunking_fakes.sha256_version(b"second"), bounds=((0.0, 2.0),))
OLD_FP = "aaaaaaaaaaaa"
NEW_FP = "bbbbbbbbbbbb"


def as_view(chunk: chunking_models.Chunk) -> models.ChunkView:
    """What the embedding step must see of a chunk the chunking step wrote."""
    return models.ChunkView(
        version=chunk.version,
        scene_set_version=chunk.scene_set_version,
        asset_version=chunk.asset_version,
        segmentation_fingerprint=chunk.segmentation_fingerprint,
        chunking_fingerprint=chunk.lineage.fingerprint,
        chunk_index=chunk.chunk_index,
        scene_indices=chunk.scene_indices,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
        text=chunk.text,
        sources=tuple(
            models.SourceView(s.source_id, s.asset_id, s.filename) for s in chunk.sources
        ),
        entitlements=chunk.entitlements,
        classifications=chunk.classifications,
    )


def as_set_view(chunk_set: chunking_models.ChunkSet) -> models.ChunkSetView:
    return models.ChunkSetView(
        version=chunk_set.version,
        scene_set_version=chunk_set.scene_set_version,
        asset_version=chunk_set.asset_version,
        chunking_fingerprint=chunk_set.lineage.fingerprint,
        chunk_count=chunk_set.chunk_count,
        created_at=chunk_set.created_at,
    )


CHUNK_SETS = [
    chunking_fakes.make_chunk_set(FIRST, OLD_FP),
    chunking_fakes.make_chunk_set(SECOND, OLD_FP),
    chunking_fakes.make_chunk_set(FIRST, NEW_FP, chunk_count=2),
]
CHUNKS = [
    *(chunking_fakes.make_chunk(FIRST, i, OLD_FP) for i in range(3)),
    chunking_fakes.make_chunk(SECOND, 0, OLD_FP),
    *(chunking_fakes.make_chunk(FIRST, i, NEW_FP, text=f"new {i}") for i in range(2)),
]


@pytest.fixture(params=["fake", pytest.param("mongo", marks=pytest.mark.integration)])
def reader(request: pytest.FixtureRequest) -> Iterator[ChunkReader]:
    if request.param == "fake":
        fake = FakeChunkReader()
        for chunk_set in CHUNK_SETS:
            fake.seed(
                [
                    as_view(c)
                    for c in CHUNKS
                    if c.lineage.fingerprint == chunk_set.lineage.fingerprint
                    and c.scene_set_version == chunk_set.scene_set_version
                ],
                as_set_view(chunk_set),
            )
        yield fake
        return
    client: MongoClient[Document] = request.getfixturevalue("mongo_client")
    database = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    # Written the way the chunking step writes them, through its own encoder.
    MongoChunkStore(client, database)
    client[database][CHUNKS_COLLECTION].insert_many([chunk_to_document(c) for c in CHUNKS])
    client[database][CHUNK_SETS_COLLECTION].insert_many(
        [chunk_set_to_document(s) for s in CHUNK_SETS]
    )
    yield MongoChunkReader(client, database)
    client.drop_database(database)


class TestGivenChunkStore:
    """Two scene sets, one of them chunked under two fingerprints."""

    def test_when_listed_under_fingerprint_then_its_sets_by_version(
        self, reader: ChunkReader
    ) -> None:
        listed = list(reader.list_chunk_sets(OLD_FP))

        assert listed == sorted((as_set_view(s) for s in CHUNK_SETS[:2]), key=lambda s: s.version)
        assert all(s.created_at.tzinfo is not None for s in listed)

    def test_when_listed_without_pin_then_every_set(self, reader: ChunkReader) -> None:
        assert {s.version for s in reader.list_chunk_sets(None)} == {s.version for s in CHUNK_SETS}

    def test_when_listed_under_unknown_fingerprint_then_nothing(self, reader: ChunkReader) -> None:
        assert list(reader.list_chunk_sets("cccccccccccc")) == []

    def test_when_chunks_fetched_then_in_index_order_under_that_fingerprint_only(
        self, reader: ChunkReader
    ) -> None:
        scene_set = FIRST[0].scene_set_version

        old = reader.list_chunks(scene_set, OLD_FP)
        new = reader.list_chunks(scene_set, NEW_FP)

        assert list(old) == [as_view(c) for c in CHUNKS[:3]]
        assert [c.text for c in new] == ["new 0", "new 1"]
        assert reader.list_chunks(scene_set, "cccccccccccc") == []
