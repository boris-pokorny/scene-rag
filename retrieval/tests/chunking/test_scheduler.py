"""One pass over the pending scene sets: derivation, dispatch and outcome counting."""

import logging

import pytest

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import ChunkingConfig, UpstreamConfig
from scene_rag.retrieval.chunking.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.retrieval.chunking.protocols import VersionRetiredError
from scene_rag.retrieval.chunking.report import ChunkingReport
from scene_rag.retrieval.chunking.scheduler import Scheduler
from tests.support.chunking_fakes import (
    DESCRIPTION_FP,
    ChunkingHarness,
    FakeSceneSetChunker,
    RecordingDispatcher,
    make_chunk_set,
    make_chunking_harness,
    make_records,
    sha256_version,
    summary_of,
)

FIRST = make_records(sha256_version(b"first"))
SECOND = make_records(sha256_version(b"second"), bounds=((0.0, 2.0),))
RECUT = make_records(sha256_version(b"first"), segmentation_fingerprint="ffffffffffff")
HALF = make_records(sha256_version(b"half"), enrichers={"description": DESCRIPTION_FP})


class TestGivenLiveSceneSets:
    """Two complete scene sets of live versions; the pass over the real service."""

    @pytest.fixture
    def chunking_harness(self, chunking_harness: ChunkingHarness) -> ChunkingHarness:
        chunking_harness.seed(FIRST, SECOND)
        return chunking_harness

    def test_when_pass_runs_then_every_set_chunked(self, chunking_harness: ChunkingHarness) -> None:
        harness = chunking_harness

        report = harness.scheduler.run_once()

        assert report == ChunkingReport(chunked=2)
        assert harness.scheduler.fingerprint == harness.service.fingerprint
        assert len(harness.store.chunks()) == 3 + 1
        assert harness.store.list_done(harness.scheduler.fingerprint) == {
            summary_of(FIRST).version,
            summary_of(SECOND).version,
        }
        assert harness.reader.listings == [models.SceneSetQuery()]

    def test_when_rerun_then_nothing_pending(self, chunking_harness: ChunkingHarness) -> None:
        harness = chunking_harness
        harness.scheduler.run_once()
        before = harness.store.chunks()
        dispatcher = RecordingDispatcher()

        report = harness.variant(dispatcher=dispatcher).run_once()

        assert report == ChunkingReport()
        assert dispatcher.batches == [[]]
        assert harness.store.chunks() == before

    def test_when_chunk_set_under_other_fingerprint_then_still_pending(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.store.put_chunk_set(make_chunk_set(FIRST, "000000000000"))

        report = harness.scheduler.run_once()

        assert report == ChunkingReport(chunked=2)
        assert len(harness.store.chunk_sets()) == 3

    def test_when_set_not_complete_then_skipped_as_not_ready(
        self, chunking_harness: ChunkingHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = chunking_harness
        harness.seed(HALF)
        caplog.set_level(logging.INFO, logger="scene_rag.retrieval.chunking.scheduler")

        report = harness.scheduler.run_once()

        assert report == ChunkingReport(chunked=2)
        assert "2 scene set(s) pending, 1 not yet complete" in caplog.text

    def test_when_asset_retired_then_its_sets_not_pending(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.reader.retire(FIRST[0].asset_version)

        report = harness.scheduler.run_once()

        assert report == ChunkingReport(chunked=1)
        assert {c.scene_set_version for c in harness.store.chunks().values()} == {
            summary_of(SECOND).version
        }

    def test_when_asset_recut_under_new_fingerprint_then_both_sets_chunked(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.seed(RECUT)

        report = harness.scheduler.run_once()

        assert report == ChunkingReport(chunked=3)

    def test_when_upstream_pinned_then_query_carries_pins_and_filter(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        config = ChunkingConfig(
            upstream=UpstreamConfig(
                segmentation_fingerprint="ffffffffffff",
                enrichment_pins=(f"description:{DESCRIPTION_FP}",),
            )
        )
        harness.seed(RECUT)

        report = harness.variant(config=config).run_once()

        assert report == ChunkingReport(chunked=1)
        assert harness.reader.listings[-1] == models.SceneSetQuery(
            "ffffffffffff", (models.EnrichmentPin("description", DESCRIPTION_FP),)
        )

    def test_when_threaded_then_same_records_as_serial(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        serial = chunking_harness
        serial.scheduler.run_once()

        with ThreadDispatcher(3) as pool:
            parallel = make_chunking_harness(FIRST, SECOND, dispatcher=pool)
            report = parallel.scheduler.run_once()

        assert report == ChunkingReport(chunked=2)
        assert parallel.store.chunks() == serial.store.chunks()
        assert parallel.store.chunk_sets() == serial.store.chunk_sets()

    def test_when_pass_logged_then_counts_and_fingerprint_named(
        self, chunking_harness: ChunkingHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="scene_rag.retrieval.chunking.scheduler")

        chunking_harness.scheduler.run_once()

        assert f"under {chunking_harness.scheduler.fingerprint}" in caplog.text
        assert "chunking pass: chunked=2" in caplog.text


class TestGivenScriptedChunker:
    """The pass over a scripted unit of work: how outcomes and failures are counted."""

    @pytest.fixture
    def chunking_harness(self, chunking_harness: ChunkingHarness) -> ChunkingHarness:
        chunking_harness.seed(FIRST, SECOND)
        return chunking_harness

    def run(self, harness: ChunkingHarness, chunker: FakeSceneSetChunker) -> ChunkingReport:
        return Scheduler(
            reader=harness.reader,
            chunk_reader=harness.store,
            chunker=chunker,
            dispatcher=SerialDispatcher(),
            config=harness.config,
        ).run_once()

    def test_when_outcomes_mixed_then_each_counted(self, chunking_harness: ChunkingHarness) -> None:
        chunker = FakeSceneSetChunker({summary_of(SECOND).version: "deferred"})

        assert self.run(chunking_harness, chunker) == ChunkingReport(chunked=1, deferred=1)
        assert sorted(chunker.calls) == sorted(
            [summary_of(FIRST).version, summary_of(SECOND).version]
        )

    @pytest.mark.parametrize(
        "error, field",
        [
            (VersionRetiredError("retired"), "dropped"),
            (RuntimeError("store down"), "failed"),
        ],
    )
    def test_when_chunker_raises_then_counted_and_pass_continues(
        self,
        chunking_harness: ChunkingHarness,
        error: Exception,
        field: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        chunker = FakeSceneSetChunker({summary_of(FIRST).version: error})
        caplog.set_level(logging.INFO, logger="scene_rag.retrieval.chunking.scheduler")

        report = self.run(chunking_harness, chunker)

        assert report == ChunkingReport(chunked=1, **{field: 1})
        assert len(chunker.calls) == 2
        assert summary_of(FIRST).version in caplog.text

    def test_when_listing_fails_then_error_propagates(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        chunking_harness.reader.failure = OSError("scene-api down")

        with pytest.raises(OSError, match="scene-api down"):
            self.run(chunking_harness, FakeSceneSetChunker())
