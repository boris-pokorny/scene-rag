"""One pass over the pending scenes: derivation, dispatch and outcome counting."""

import logging
from typing import cast

import pytest

from scene_rag.extraction.enrichment.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.extraction.enrichment.models import EnrichmentReport, SceneSetSummary
from scene_rag.extraction.enrichment.protocols import VersionRetiredError
from scene_rag.extraction.enrichment.scheduler import Scheduler, Work
from tests.support.enrichment_fakes import (
    EnrichmentHarness,
    FakeSceneSetEnricher,
    RecordingDispatcher,
    make_enrichment,
    make_scene_set,
    sha256_version,
)

FIRST = make_scene_set(sha256_version(b"first"))
SECOND = make_scene_set(sha256_version(b"second"), bounds=((0.0, 2.0),))
RECUT = make_scene_set(sha256_version(b"first"), segmentation_fingerprint="ffffffffffff")


class FailingAssetReader:
    """AssetVersionReader whose listing raises."""

    def get(self, version: str) -> None:
        return None

    def list_live_versions(self) -> frozenset[str]:
        raise OSError("registry down")


class TestGivenLiveSceneSets:
    """Two scene sets of live versions; the pass over the real service."""

    @pytest.fixture
    def enrichment_harness(self, enrichment_harness: EnrichmentHarness) -> EnrichmentHarness:
        enrichment_harness.seed(FIRST, SECOND)
        return enrichment_harness

    def test_when_pass_runs_then_every_scene_enriched_by_every_enricher(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness

        report = harness.scheduler.run_once()

        assert report == EnrichmentReport(enriched=2)
        assert harness.scheduler.fingerprints == harness.service.fingerprints
        assert len(harness.store.snapshot()) == (3 + 1) * 2
        for name, fingerprint in harness.scheduler.fingerprints.items():
            assert harness.store.list_done(name, fingerprint) == {
                (FIRST.version, 0),
                (FIRST.version, 1),
                (FIRST.version, 2),
                (SECOND.version, 0),
            }

    def test_when_rerun_then_nothing_pending(self, enrichment_harness: EnrichmentHarness) -> None:
        harness = enrichment_harness
        harness.scheduler.run_once()
        before = harness.store.snapshot()
        dispatcher = RecordingDispatcher()

        report = harness.variant(dispatcher=dispatcher).run_once()

        assert report == EnrichmentReport()
        assert dispatcher.batches == [[]]
        assert harness.store.snapshot() == before

    def test_when_some_records_exist_then_only_missing_scenes_pending(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        fingerprints = harness.service.fingerprints
        harness.store.put(make_enrichment(FIRST, 0, "description", fingerprints["description"]))
        harness.store.put(make_enrichment(FIRST, 1, "description", fingerprints["description"]))
        harness.store.put(make_enrichment(SECOND, 0, "transcript", fingerprints["transcript"]))
        dispatcher = RecordingDispatcher()

        report = harness.variant(dispatcher=dispatcher).run_once()

        assert report == EnrichmentReport(enriched=2)
        (batch,) = dispatcher.batches
        work = cast("list[Work]", batch)
        assert {summary.version: dict(pending) for summary, pending in work} == {
            FIRST.version: {"description": frozenset({2}), "transcript": frozenset({0, 1, 2})},
            SECOND.version: {"description": frozenset({0})},
        }
        assert len(harness.store.snapshot()) == 8

    def test_when_records_under_other_fingerprint_then_still_pending(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        for index in range(3):
            harness.store.put(make_enrichment(FIRST, index, "description", "000000000000"))

        report = harness.scheduler.run_once()

        assert report == EnrichmentReport(enriched=2)
        assert len(harness.store.snapshot()) == 3 + 8

    def test_when_asset_retired_then_its_sets_not_pending(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        harness.assets.retire(FIRST.asset_version)

        report = harness.scheduler.run_once()

        assert report == EnrichmentReport(enriched=1)
        assert {r.scene_set_version for r in harness.store.snapshot().values()} == {SECOND.version}

    def test_when_asset_recut_under_new_fingerprint_then_both_sets_enriched(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        harness.seed(RECUT)

        report = harness.scheduler.run_once()

        assert report == EnrichmentReport(enriched=3)
        assert {r.scene_set_version for r in harness.store.snapshot().values()} == {
            FIRST.version,
            SECOND.version,
            RECUT.version,
        }

    def test_when_threaded_then_same_records_as_serial(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        serial = enrichment_harness
        serial.scheduler.run_once()
        threaded = enrichment_harness.__class__
        del threaded
        from tests.support.enrichment_fakes import make_enrichment_harness

        with ThreadDispatcher(3) as pool:
            parallel = make_enrichment_harness(FIRST, SECOND, dispatcher=pool)
            report = parallel.scheduler.run_once()

        assert report == EnrichmentReport(enriched=2)
        assert parallel.store.snapshot() == serial.store.snapshot()

    def test_when_pass_logged_then_counts_and_fingerprints_named(
        self, enrichment_harness: EnrichmentHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="scene_rag.extraction.enrichment.scheduler")

        enrichment_harness.scheduler.run_once()

        assert "2 scene set(s), 8 scene enrichment(s) pending" in caplog.text
        assert "enrichment pass: enriched=2" in caplog.text


class TestGivenScriptedEnricher:
    """The pass over a scripted unit of work: how outcomes and failures are counted."""

    @pytest.fixture
    def enrichment_harness(self, enrichment_harness: EnrichmentHarness) -> EnrichmentHarness:
        enrichment_harness.seed(FIRST, SECOND)
        return enrichment_harness

    def run(self, harness: EnrichmentHarness, enricher: FakeSceneSetEnricher) -> EnrichmentReport:
        return Scheduler(
            assets=harness.assets,
            scene_reader=harness.scenes,
            enrichment_reader=harness.store,
            enricher=enricher,
            dispatcher=SerialDispatcher(),
        ).run_once()

    def test_when_outcomes_mixed_then_each_counted(
        self, enrichment_harness: EnrichmentHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        enricher = FakeSceneSetEnricher({SECOND.version: "deferred"})

        assert self.run(enrichment_harness, enricher) == EnrichmentReport(enriched=1, deferred=1)
        assert sorted(v for v, _ in enricher.calls) == sorted([FIRST.version, SECOND.version])
        assert dict(enricher.calls) == {
            FIRST.version: {"description": frozenset({0, 1, 2})},
            SECOND.version: {"description": frozenset({0})},
        }

    @pytest.mark.parametrize(
        "error, field",
        [
            (VersionRetiredError("retired"), "dropped"),
            (RuntimeError("store down"), "failed"),
        ],
    )
    def test_when_enricher_raises_then_counted_and_pass_continues(
        self,
        enrichment_harness: EnrichmentHarness,
        error: Exception,
        field: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        enricher = FakeSceneSetEnricher({FIRST.version: error})
        caplog.set_level(logging.INFO, logger="scene_rag.extraction.enrichment.scheduler")

        report = self.run(enrichment_harness, enricher)

        assert report == EnrichmentReport(enriched=1, **{field: 1})
        assert len(enricher.calls) == 2
        assert FIRST.version in caplog.text

    def test_when_listing_fails_then_error_propagates(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        scheduler = Scheduler(
            assets=FailingAssetReader(),
            scene_reader=enrichment_harness.scenes,
            enrichment_reader=enrichment_harness.store,
            enricher=FakeSceneSetEnricher(),
            dispatcher=SerialDispatcher(),
        )

        with pytest.raises(OSError, match="registry down"):
            scheduler.run_once()

    def test_when_summary_counts_scenes_then_indices_derived(self) -> None:
        summary = SceneSetSummary("sha256:aa", "sha256:bb", 2)

        assert summary.scene_count == 2
