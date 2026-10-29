"""One pass over the pending chunk sets: derivation, dispatch and outcome counting."""

import logging

import pytest

from scene_rag.retrieval.embedding.config import EmbeddingConfig, UpstreamConfig
from scene_rag.retrieval.embedding.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.retrieval.embedding.report import EmbeddingReport
from scene_rag.retrieval.embedding.scheduler import Scheduler
from tests.support.embedding_fakes import (
    EmbeddingHarness,
    FakeChunkSetEmbedder,
    RecordingDispatcher,
    make_chunk_set_view,
    make_chunk_views,
    make_embedding_harness,
    make_index_set,
    sha256_version,
)

FIRST = make_chunk_views(sha256_version(b"first"))
SECOND = make_chunk_views(sha256_version(b"second"), count=1)
RECHUNKED = make_chunk_views(sha256_version(b"first"), chunking_fingerprint="ffffffffffff")


class TestGivenChunkSets:
    """Two chunk sets; the pass over the real service."""

    @pytest.fixture
    def embedding_harness(self, embedding_harness: EmbeddingHarness) -> EmbeddingHarness:
        embedding_harness.seed(FIRST, SECOND)
        return embedding_harness

    def test_when_pass_runs_then_every_set_indexed(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness

        report = harness.scheduler.run_once()

        assert report == EmbeddingReport(indexed=2)
        assert harness.scheduler.fingerprint == harness.service.fingerprint
        assert harness.search_index.count(harness.index.collection) == 4
        assert harness.store.list_done(harness.scheduler.fingerprint) == {
            make_chunk_set_view(FIRST).version,
            make_chunk_set_view(SECOND).version,
        }
        assert harness.reader.listings == [None]

    def test_when_rerun_then_nothing_pending(self, embedding_harness: EmbeddingHarness) -> None:
        harness = embedding_harness
        harness.scheduler.run_once()
        before = harness.search_index.points(harness.index.collection)
        dispatcher = RecordingDispatcher()

        report = harness.variant(dispatcher=dispatcher).run_once()

        assert report == EmbeddingReport()
        assert dispatcher.batches == [[]]
        assert harness.search_index.points(harness.index.collection) == before

    def test_when_index_set_under_other_fingerprint_then_still_pending(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        harness.store.put_index_set(make_index_set(FIRST, fingerprint="000000000000"))

        report = harness.scheduler.run_once()

        assert report == EmbeddingReport(indexed=2)
        assert len(harness.store.index_sets()) == 3

    def test_when_rechunked_under_new_fingerprint_then_both_sets_indexed(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        harness.seed(RECHUNKED)

        report = harness.scheduler.run_once()

        assert report == EmbeddingReport(indexed=3)
        assert harness.search_index.count(harness.index.collection) == 7

    def test_when_chunking_pinned_then_only_that_fingerprint_listed(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        harness = embedding_harness
        harness.seed(RECHUNKED)
        config = EmbeddingConfig(upstream=UpstreamConfig(chunking_fingerprint="ffffffffffff"))

        report = harness.variant(config=config).run_once()

        assert report == EmbeddingReport(indexed=1)
        assert harness.reader.listings[-1] == "ffffffffffff"

    def test_when_threaded_then_same_records_as_serial(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        serial = embedding_harness
        serial.scheduler.run_once()

        with ThreadDispatcher(3) as pool:
            parallel = make_embedding_harness(FIRST, SECOND, dispatcher=pool)
            report = parallel.scheduler.run_once()

        assert report == EmbeddingReport(indexed=2)
        assert parallel.search_index.points(
            parallel.index.collection
        ) == serial.search_index.points(serial.index.collection)
        assert parallel.store.index_sets() == serial.store.index_sets()

    def test_when_pass_logged_then_counts_and_fingerprint_named(
        self, embedding_harness: EmbeddingHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="scene_rag.retrieval.embedding.scheduler")

        embedding_harness.scheduler.run_once()

        assert (
            f"2 chunk set(s) pending under {embedding_harness.scheduler.fingerprint}" in caplog.text
        )
        assert "embedding pass: indexed=2" in caplog.text


class TestGivenScriptedEmbedder:
    """The pass over a scripted unit of work: how outcomes and failures are counted."""

    @pytest.fixture
    def embedding_harness(self, embedding_harness: EmbeddingHarness) -> EmbeddingHarness:
        embedding_harness.seed(FIRST, SECOND)
        return embedding_harness

    def run(self, harness: EmbeddingHarness, embedder: FakeChunkSetEmbedder) -> EmbeddingReport:
        return Scheduler(
            chunk_reader=harness.reader,
            store_reader=harness.store,
            embedder=embedder,
            dispatcher=SerialDispatcher(),
            config=harness.config,
        ).run_once()

    def test_when_outcomes_mixed_then_each_counted(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        embedder = FakeChunkSetEmbedder({make_chunk_set_view(SECOND).version: "deferred"})

        assert self.run(embedding_harness, embedder) == EmbeddingReport(indexed=1, deferred=1)
        assert sorted(embedder.calls) == sorted(
            [make_chunk_set_view(FIRST).version, make_chunk_set_view(SECOND).version]
        )

    def test_when_embedder_raises_then_counted_failed_and_pass_continues(
        self, embedding_harness: EmbeddingHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = make_chunk_set_view(FIRST).version
        embedder = FakeChunkSetEmbedder({first: RuntimeError("index down")})
        caplog.set_level(logging.INFO, logger="scene_rag.retrieval.embedding.scheduler")

        report = self.run(embedding_harness, embedder)

        assert report == EmbeddingReport(indexed=1, failed=1)
        assert len(embedder.calls) == 2
        assert first in caplog.text

    def test_when_listing_fails_then_error_propagates(
        self, embedding_harness: EmbeddingHarness
    ) -> None:
        embedding_harness.reader.failure = OSError("store down")

        with pytest.raises(OSError, match="store down"):
            self.run(embedding_harness, FakeChunkSetEmbedder())
