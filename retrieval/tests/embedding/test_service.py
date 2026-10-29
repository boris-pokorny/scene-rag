"""The per-chunk-set unit of work: chunks in, points and an index set out."""

import dataclasses
from collections.abc import Sequence

import pytest

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.protocols import ModelError
from tests.support.embedding_fakes import (
    T0,
    EmbeddingHarness,
    FakeEmbeddingModel,
    RecordingWriter,
    fake_vector,
    make_chunk_set_view,
    make_chunk_views,
    make_index,
    sha256_version,
)

CHUNKS = make_chunk_views(entitlements=frozenset({"public", "team-platform"}))
CHUNK_SET = make_chunk_set_view(CHUNKS)


class TestGivenChunkSet:
    """One complete chunk set of three chunks, nothing embedded yet; batches of two."""

    @pytest.fixture
    def embedding_harness(self, embedding_harness: EmbeddingHarness) -> EmbeddingHarness:
        embedding_harness.seed(CHUNKS)
        return embedding_harness

    def test_when_embedded_then_batches_upserted_then_index_set(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness

        outcome = harness.service.embed(CHUNK_SET)

        assert outcome == "indexed"
        assert harness.writer.commands == [
            ("upsert", CHUNK_SET.version, 2),
            ("upsert", CHUNK_SET.version, 1),
            ("put_index_set", CHUNK_SET.version, 3),
        ]
        assert harness.model.batches == [("chunk 0 text", "chunk 1 text"), ("chunk 2 text",)]
        assert harness.reader.fetches == [
            (CHUNK_SET.scene_set_version, CHUNK_SET.chunking_fingerprint)
        ]

    def test_when_embedded_then_point_carries_vector_identity_and_provenance(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        fingerprint = harness.service.fingerprint

        harness.service.embed(CHUNK_SET)

        chunk = CHUNKS[1]
        point = harness.search_index.points(harness.index.collection)[
            models.point_id(chunk.version)
        ]
        version = models.index_entry_version(chunk.version, fingerprint, "sha256")
        assert point.vector == fake_vector(chunk.text)
        assert point.entry == models.IndexEntry(
            version=version,
            chunk_version=chunk.version,
            chunk_set_version=CHUNK_SET.version,
            scene_set_version=chunk.scene_set_version,
            asset_version=chunk.asset_version,
            chunking_fingerprint=chunk.chunking_fingerprint,
            segmentation_fingerprint=chunk.segmentation_fingerprint,
            chunk_index=1,
            scene_indices=(1,),
            start_seconds=4.0,
            end_seconds=8.0,
            text="chunk 1 text",
            sources=chunk.sources,
            entitlements=frozenset({"public", "team-platform"}),
            classifications=("internal",),
            created_at=T0,
            lineage=models.Lineage(
                fingerprint=fingerprint, inputs=(f"chunk:{chunk.version}",), version=version
            ),
        )

    def test_when_embedded_then_index_set_names_every_chunk(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        fingerprint = harness.service.fingerprint

        harness.service.embed(CHUNK_SET)

        version = models.index_set_version(CHUNK_SET.version, fingerprint, "sha256")
        assert harness.store.get_index_set(version) == models.IndexSet(
            version=version,
            chunk_set_version=CHUNK_SET.version,
            scene_set_version=CHUNK_SET.scene_set_version,
            asset_version=CHUNK_SET.asset_version,
            chunking_fingerprint=CHUNK_SET.chunking_fingerprint,
            collection=harness.index.collection,
            vector_count=3,
            created_at=T0,
            lineage=models.Lineage(
                fingerprint=fingerprint,
                inputs=tuple(f"chunk:{c.version}" for c in CHUNKS),
                version=version,
            ),
        )
        assert harness.store.list_done(fingerprint) == {CHUNK_SET.version}

    def test_when_embedded_again_then_duplicate_and_marker_unchanged(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        harness.service.embed(CHUNK_SET)
        before = harness.store.index_sets()
        harness.clock.tick()

        outcome = harness.service.embed(CHUNK_SET)

        assert outcome == "duplicate"
        assert harness.store.index_sets() == before
        assert harness.search_index.count(harness.index.collection) == 3

    def test_when_worker_died_after_points_then_rerun_completes_the_set(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        dying = RecordingWriter(harness.search_index, harness.store, fail_after=2)
        with pytest.raises(RuntimeError, match="put_index_set"):
            harness.variant_service(writer=dying).embed(CHUNK_SET)
        assert harness.store.index_sets() == {}
        assert harness.search_index.count(harness.index.collection) == 3

        outcome = harness.service.embed(CHUNK_SET)

        assert outcome == "indexed"
        assert harness.search_index.count(harness.index.collection) == 3
        assert len(harness.store.index_sets()) == 1

    def test_when_batch_size_covers_the_set_then_one_request(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness

        harness.variant_service(batch_size=10).embed(CHUNK_SET)

        assert harness.model.batches == [tuple(c.text for c in CHUNKS)]

    def test_when_model_fails_then_error_propagates_before_any_write(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness

        with pytest.raises(ModelError, match="model down"):
            harness.variant_service(model=FakeEmbeddingModel(fail=True)).embed(CHUNK_SET)

        assert harness.writer.commands == []

    def test_when_vectors_have_other_dimensions_then_error_before_any_write(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        short = FakeEmbeddingModel(dimensions=4)

        with pytest.raises(ModelError, match="4 dimensions"):
            harness.variant_service(model=short).embed(CHUNK_SET)

        assert harness.writer.commands == []

    def test_when_model_answers_wrong_count_then_error(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness

        class Short(FakeEmbeddingModel):
            def embed(self, texts: Sequence[str]) -> tuple[models.Vector, ...]:
                return super().embed(texts)[:-1]

        with pytest.raises(ModelError, match="1 vector\\(s\\) for 2 chunk"):
            harness.variant_service(model=Short()).embed(CHUNK_SET)


class TestGivenEmptyChunkSet:
    """A chunk set with no chunks: no scene had text."""

    def test_when_embedded_then_index_set_with_zero_vectors_and_no_upsert(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        empty = make_chunk_set_view(asset_version=sha256_version(b"silent"))
        harness.reader.seed([], empty)

        outcome = harness.service.embed(empty)

        assert outcome == "indexed"
        assert harness.writer.commands == [("put_index_set", empty.version, 0)]
        (index_set,) = harness.store.index_sets().values()
        assert index_set.vector_count == 0 and index_set.lineage.inputs == ()


class TestGivenChangedChunkSet:
    """The chunks differ from the listing by the time they are fetched."""

    @pytest.fixture
    def embedding_harness(self, embedding_harness: EmbeddingHarness) -> EmbeddingHarness:
        embedding_harness.seed(CHUNKS)
        return embedding_harness

    def test_when_fewer_chunks_than_listed_then_deferred(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        harness.reader.chunks[CHUNK_SET.version] = CHUNKS[:2]

        assert harness.service.embed(CHUNK_SET) == "deferred"
        assert harness.writer.commands == [] and harness.model.batches == []

    def test_when_index_out_of_sequence_then_deferred(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        harness.reader.chunks[CHUNK_SET.version] = (
            CHUNKS[0],
            dataclasses.replace(CHUNKS[1], chunk_index=5),
            CHUNKS[2],
        )

        assert harness.service.embed(CHUNK_SET) == "deferred"


class TestGivenConstruction:
    """What the service refuses to be built with."""

    def test_when_batch_size_below_one_then_value_error(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            embedding_harness.variant_service(batch_size=0)

    def test_when_index_is_not_the_models_then_value_error(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        with pytest.raises(ValueError, match="fingerprint"):
            embedding_harness.variant_service(index=make_index("000000000000"))
