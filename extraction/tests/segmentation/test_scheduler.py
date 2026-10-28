"""One pass over the pending versions: derivation, dispatch and outcome counting."""

import logging
import threading
from datetime import timedelta
from pathlib import Path
from typing import Literal

import pytest

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.extraction.segmentation.models import SegmentationReport
from scene_rag.extraction.segmentation.protocols import (
    Analysis,
    Dispatcher,
    MediaUnavailableError,
    VersionRetiredError,
)
from scene_rag.extraction.segmentation.scheduler import Scheduler
from tests.support.segmentation_fakes import (
    T0,
    FakeAssetVersionReader,
    FakeAudioClipper,
    FakeDerivativeStore,
    FakeKeyframeRenderer,
    FakeLocator,
    FakeSceneStore,
    FakeSegmenter,
    FakeSegmentExtractor,
    RecordingDispatcher,
    RecordingSceneWriter,
    SegmentationHarness,
    make_scene_set,
    make_segmentation_harness,
    make_view,
    sha256_version,
)

VERSION = sha256_version(b"hello")
OTHER = sha256_version(b"other")
FINGERPRINT = "0123456789ab"


class FailingAssetReader:
    """AssetVersionReader whose listing raises."""

    def get(self, version: str) -> models.AssetVersionView | None:
        return None

    def list_live(self) -> list[models.AssetVersionView]:
        raise OSError("registry down")


class TestGivenLiveVersion:
    """One live version registered, nothing segmented yet; the pass over the real service."""

    @pytest.fixture
    def segmentation_harness(
        self, segmentation_harness: SegmentationHarness
    ) -> SegmentationHarness:
        segmentation_harness.assets.records[VERSION] = make_view(VERSION)
        return segmentation_harness

    def test_when_pass_runs_then_scene_set_written(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness

        report = harness.scheduler.run_once()

        assert report == SegmentationReport(segmented=1)
        assert harness.scheduler.fingerprint == harness.service.fingerprint
        (scene_set,) = harness.scene_store.snapshot().values()
        assert scene_set.asset_version == VERSION
        assert scene_set.lineage.fingerprint == harness.scheduler.fingerprint
        assert harness.locator.located == [VERSION]

    def test_when_rerun_then_nothing_written(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        harness.scheduler.run_once()
        before = harness.scene_store.snapshot()
        harness.writer.commands.clear()
        harness.clock.tick()

        report = harness.scheduler.run_once()

        assert report == SegmentationReport()
        assert harness.writer.commands == []
        assert harness.scene_store.snapshot() == before

    def test_when_fingerprint_changes_then_new_set_next_to_old(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        harness.scheduler.run_once()
        retuned = harness.variant(config=SegmentationConfig(scene_threshold=30.0))
        assert retuned.fingerprint != harness.scheduler.fingerprint

        report = retuned.run_once()

        assert report == SegmentationReport(segmented=1)
        fingerprints = {s.lineage.fingerprint for s in harness.scene_store.snapshot().values()}
        assert fingerprints == {harness.scheduler.fingerprint, retuned.fingerprint}
        assert harness.variant().run_once() == SegmentationReport()

    def test_when_source_unreadable_then_deferred_and_nothing_written(
        self, segmentation_harness: SegmentationHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = segmentation_harness

        report = harness.variant(locator=FakeLocator(available=False)).run_once()

        assert report == SegmentationReport(deferred=1)
        assert harness.writer.commands == []
        assert harness.derivatives.puts == []
        assert f"deferred {VERSION}" in caplog.text

    def test_when_retired_meanwhile_then_dropped_before_write(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness

        class RetiringExtractor(FakeSegmentExtractor):
            def extract(self, path: Path) -> Analysis:
                harness.assets.retire(VERSION, at=T0 + timedelta(seconds=1))
                return super().extract(path)

        report = harness.variant(extractor=RetiringExtractor()).run_once()

        assert report == SegmentationReport(dropped=1)
        assert harness.writer.commands == []
        assert harness.scene_store.snapshot() == {}

    def test_when_another_worker_wrote_first_then_duplicate(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness

        class RacingExtractor(FakeSegmentExtractor):
            def extract(self, path: Path) -> Analysis:
                harness.variant(extractor=FakeSegmentExtractor()).run_once()
                return super().extract(path)

        report = harness.variant(extractor=RacingExtractor()).run_once()

        assert report == SegmentationReport(duplicate=1)
        assert len(harness.scene_store.snapshot()) == 1

    @pytest.mark.parametrize(
        "broken",
        [
            {"extractor": FakeSegmentExtractor(fail=True)},
            {"keyframes": FakeKeyframeRenderer(fail=True)},
            {"audio": FakeAudioClipper(fail=True)},
            {"derivatives": FakeDerivativeStore(fail=True)},
        ],
        ids=["extractor", "keyframes", "audio", "derivatives"],
    )
    def test_when_collaborator_fails_then_counted_and_nothing_written(
        self,
        segmentation_harness: SegmentationHarness,
        broken: dict[str, object],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        harness = segmentation_harness

        report = harness.variant(**broken).run_once()  # type: ignore[arg-type]

        assert report == SegmentationReport(failed=1)
        assert harness.writer.commands == []
        assert f"failed to segment {VERSION}" in caplog.text

    def test_when_writer_fails_then_counted(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        failing = RecordingSceneWriter(harness.scene_store, fail=True)

        report = harness.variant(writer=failing).run_once()

        assert report == SegmentationReport(failed=1)
        assert failing.commands == [("put", VERSION)]
        assert harness.scene_store.snapshot() == {}


class TestGivenMixedRegistry:
    """A retired version, a live one and a failing one."""

    def test_when_pass_runs_then_only_live_versions_considered_and_pass_continues(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        harness.assets.records[OTHER] = make_view(OTHER, retired_at=T0)
        harness.assets.records[VERSION] = make_view(VERSION)
        broken = sha256_version(b"broken")
        harness.assets.records[broken] = make_view(broken)
        locator = FakeLocator()

        class SelectiveExtractor(FakeSegmentExtractor):
            def extract(self, path: Path) -> Analysis:
                if locator.located[-1] == broken:
                    raise OSError("decoder crashed")
                return super().extract(path)

        report = harness.variant(locator=locator, extractor=SelectiveExtractor()).run_once()

        assert report == SegmentationReport(segmented=1, failed=1)
        assert harness.scene_store.list_asset_versions(harness.scheduler.fingerprint) == {VERSION}

    def test_when_listing_fails_then_error_propagates(self) -> None:
        harness = make_segmentation_harness()
        scheduler = Scheduler(
            assets=FailingAssetReader(),  # the listing is the one thing a pass cannot skip
            scene_reader=harness.scene_store,
            segmenter=harness.service,
            dispatcher=SerialDispatcher(),
        )

        with pytest.raises(OSError, match="registry down"):
            scheduler.run_once()


def scripted(
    versions: dict[str, Literal["segmented", "duplicate"] | Exception],
    *,
    done: tuple[str, ...] = (),
    retired: tuple[str, ...] = (),
    dispatcher: Dispatcher | None = None,
) -> tuple[Scheduler, FakeSegmenter, RecordingDispatcher]:
    """A scheduler over a scripted segmenter; ``done`` versions already have a scene set."""
    assets = FakeAssetVersionReader(
        *(make_view(v) for v in (*versions, *done)),
        *(make_view(v, retired_at=T0) for v in retired),
    )
    scene_store = FakeSceneStore()
    for version in done:
        scene_store.put(make_scene_set(asset_version=version, fingerprint=FINGERPRINT))
    segmenter = FakeSegmenter(versions, fingerprint=FINGERPRINT)
    recording = RecordingDispatcher(dispatcher)
    scheduler = Scheduler(
        assets=assets, scene_reader=scene_store, segmenter=segmenter, dispatcher=recording
    )
    return scheduler, segmenter, recording


class TestGivenScriptedSegmenter:
    """The scheduler alone: what it dispatches and how it counts what comes back."""

    def test_when_outcomes_mixed_then_each_counted_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        written, duplicate, unreadable, retired, crashed = (
            sha256_version(name) for name in (b"a", b"b", b"c", b"d", b"e")
        )
        scheduler, segmenter, _ = scripted(
            {
                written: "segmented",
                duplicate: "duplicate",
                unreadable: MediaUnavailableError("no readable source"),
                retired: VersionRetiredError(retired),
                crashed: OSError("decoder crashed"),
            }
        )

        caplog.set_level(logging.INFO, logger=Scheduler.__module__)

        report = scheduler.run_once()

        assert report == SegmentationReport(
            segmented=1, duplicate=1, dropped=1, deferred=1, failed=1
        )
        assert sorted(segmenter.calls) == sorted([written, duplicate, unreadable, retired, crashed])
        assert f"deferred {unreadable}: no readable source" in caplog.text
        assert f"dropped {retired}: retired while segmenting" in caplog.text
        assert f"failed to segment {crashed}" in caplog.text
        assert "decoder crashed" in caplog.text

    def test_when_some_versions_done_or_retired_then_only_the_rest_dispatched(self) -> None:
        pending = sha256_version(b"pending")
        scheduler, segmenter, dispatcher = scripted(
            {pending: "segmented"},
            done=(sha256_version(b"done"),),
            retired=(sha256_version(b"gone"),),
        )

        report = scheduler.run_once()

        assert report == SegmentationReport(segmented=1)
        assert segmenter.calls == [pending]
        assert [[r.version for r in batch] for batch in dispatcher.batches] == [[pending]]  # type: ignore[attr-defined]

    def test_when_nothing_pending_then_empty_batch_and_zero_report(self) -> None:
        scheduler, segmenter, dispatcher = scripted({}, done=(sha256_version(b"done"),))

        assert scheduler.run_once() == SegmentationReport()
        assert segmenter.calls == []
        assert dispatcher.batches == [[]]

    def test_when_dispatched_on_threads_then_same_report_as_serial(self) -> None:
        versions = {sha256_version(bytes([i])): "segmented" for i in range(6)}
        script: dict[str, Literal["segmented", "duplicate"] | Exception] = {
            **versions,  # type: ignore[dict-item]
            sha256_version(b"x"): OSError("boom"),
            sha256_version(b"y"): MediaUnavailableError("gone"),
        }
        serial, _, _ = scripted(script)
        with ThreadDispatcher(3) as threads:
            threaded, segmenter, _ = scripted(script, dispatcher=threads)

            report = threaded.run_once()

        assert report == serial.run_once() == SegmentationReport(segmented=6, deferred=1, failed=1)
        assert sorted(segmenter.calls) == sorted(script)


class TestGivenTwoWorkers:
    """A dispatcher of two threads over two versions that each wait for the other."""

    @pytest.fixture
    def harness(self) -> SegmentationHarness:
        return make_segmentation_harness(make_view(VERSION), make_view(OTHER))

    @staticmethod
    def meeting_extractor(parties: int, timeout: float = 5) -> FakeSegmentExtractor:
        barrier = threading.Barrier(parties, timeout=timeout)

        class MeetingExtractor(FakeSegmentExtractor):
            def extract(self, path: Path) -> Analysis:
                barrier.wait()
                return super().extract(path)

        return MeetingExtractor()

    def test_when_two_threads_then_both_versions_in_flight_at_once(
        self, harness: SegmentationHarness
    ) -> None:
        with ThreadDispatcher(2) as threads:
            scheduler = harness.variant(extractor=self.meeting_extractor(2), dispatcher=threads)

            report = scheduler.run_once()

        assert report == SegmentationReport(segmented=2)
        assert harness.scene_store.list_asset_versions(scheduler.fingerprint) == {VERSION, OTHER}

    def test_when_serial_then_versions_never_meet(self, harness: SegmentationHarness) -> None:
        scheduler = harness.variant(extractor=self.meeting_extractor(2, timeout=0.2))

        report = scheduler.run_once()

        assert report == SegmentationReport(failed=2)  # each waited alone and timed out
        assert harness.scene_store.snapshot() == {}
