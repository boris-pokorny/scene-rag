from datetime import timedelta
from pathlib import Path

import pytest

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.models import SegmentationReport
from scene_rag.extraction.segmentation.protocols import Analysis
from tests.support.segmentation_fakes import (
    T0,
    FakeAudioClipper,
    FakeDerivativeStore,
    FakeKeyframeRenderer,
    FakeLocator,
    FakeSegmentExtractor,
    RecordingSceneWriter,
    SegmentationHarness,
    make_analysis,
    make_segmentation_harness,
    make_view,
    sha256_version,
)

VERSION = sha256_version(b"hello")
OTHER = sha256_version(b"other")


class FailingAssetReader:
    """AssetVersionReader whose listing raises."""

    def get(self, version: str) -> models.AssetVersionView | None:
        return None

    def list_live(self) -> list[models.AssetVersionView]:
        raise OSError("registry down")


class TestGivenLiveVersion:
    """One live version registered, nothing segmented yet."""

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

        report = harness.service.run_once()

        assert report == SegmentationReport(segmented=1)
        (scene_set,) = harness.scene_store.snapshot().values()
        assert scene_set.asset_version == VERSION
        assert scene_set.version == models.scene_set_version(
            VERSION, harness.service.fingerprint, "sha256"
        )
        assert scene_set.lineage == models.Lineage(
            fingerprint=harness.service.fingerprint,
            inputs=(f"asset-version:{VERSION}",),
            version=scene_set.version,
        )
        assert scene_set.created_at == T0
        assert scene_set.media == make_analysis().media
        assert [s.index for s in scene_set.scenes] == [0, 1, 2]
        assert scene_set.scenes[2].end_seconds == 12.5
        assert harness.locator.located == [VERSION]
        assert harness.extractor.calls == [harness.locator.path]

    def test_when_pass_runs_then_keyframes_and_clips_filed_by_content(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness

        harness.service.run_once()

        (scene_set,) = harness.scene_store.snapshot().values()
        first = scene_set.scenes[0]
        assert [k.frame for k in first.keyframes] == [0, 10]
        assert first.keyframes[1].seconds == 0.4
        assert first.keyframes[0].derivative.media_type == "image/jpeg"
        assert harness.derivatives.blobs[first.keyframes[0].derivative.digest] == b"jpeg:0:0"
        assert first.audio_clip is not None
        assert first.audio_clip.media_type == "audio/wav"
        assert harness.derivatives.blobs[first.audio_clip.digest] == b"wav:0"
        assert len(harness.derivatives.puts) == 3 * 2 + 3

    def test_when_renderer_repeats_a_frame_then_filed_once(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness

        harness.variant(keyframes=FakeKeyframeRenderer(duplicate_frames=True)).run_once()

        (scene_set,) = harness.scene_store.snapshot().values()
        assert [k.frame for k in scene_set.scenes[0].keyframes] == [0, 0]
        assert len({k.derivative.digest for k in scene_set.scenes[0].keyframes}) == 1
        assert len(harness.derivatives.blobs) == 3 + 3

    def test_when_no_audio_stream_then_clips_absent(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness

        report = harness.variant(audio=FakeAudioClipper(silent=True)).run_once()

        assert report == SegmentationReport(segmented=1)
        (scene_set,) = harness.scene_store.snapshot().values()
        assert all(scene.audio_clip is None for scene in scene_set.scenes)

    def test_when_video_has_no_cuts_then_one_scene(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        whole: Analysis = make_analysis(bounds=((0.0, 12.5),))

        harness.variant(extractor=FakeSegmentExtractor(whole)).run_once()

        (scene_set,) = harness.scene_store.snapshot().values()
        assert len(scene_set.scenes) == 1
        assert (scene_set.scenes[0].start_frame, scene_set.scenes[0].end_frame) == (0, 312)

    def test_when_rerun_then_nothing_written(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        harness.service.run_once()
        before = harness.scene_store.snapshot()
        harness.writer.commands.clear()
        harness.clock.tick()

        report = harness.service.run_once()

        assert report == SegmentationReport()
        assert harness.writer.commands == []
        assert harness.scene_store.snapshot() == before

    def test_when_fingerprint_changes_then_new_set_next_to_old(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        harness.service.run_once()
        retuned = harness.variant(config=SegmentationConfig(scene_threshold=30.0))
        assert retuned.fingerprint != harness.service.fingerprint

        report = retuned.run_once()

        assert report == SegmentationReport(segmented=1)
        fingerprints = {s.lineage.fingerprint for s in harness.scene_store.snapshot().values()}
        assert fingerprints == {harness.service.fingerprint, retuned.fingerprint}
        assert harness.variant().run_once() == SegmentationReport()

    def test_when_source_unreadable_then_deferred_and_nothing_written(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness

        report = harness.variant(locator=FakeLocator(available=False)).run_once()

        assert report == SegmentationReport(deferred=1)
        assert harness.writer.commands == []
        assert harness.derivatives.puts == []

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
        self, segmentation_harness: SegmentationHarness, broken: dict[str, object]
    ) -> None:
        harness = segmentation_harness

        report = harness.variant(**broken).run_once()  # type: ignore[arg-type]

        assert report == SegmentationReport(failed=1)
        assert harness.writer.commands == []

    def test_when_writer_fails_then_counted(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        failing = RecordingSceneWriter(harness.scene_store, fail=True)

        report = harness.variant(writer=failing).run_once()

        assert report == SegmentationReport(failed=1)
        assert failing.commands == [("put", VERSION)]
        assert harness.scene_store.snapshot() == {}

    def test_when_workspace_root_given_then_scratch_files_live_under_it_and_are_removed(
        self, segmentation_harness: SegmentationHarness, tmp_path: Path
    ) -> None:
        harness = segmentation_harness
        service = harness.variant(workspace_root=tmp_path)

        service.run_once()

        assert harness.derivatives.paths
        assert all(path.is_relative_to(tmp_path) for path in harness.derivatives.paths)
        assert list(tmp_path.iterdir()) == []

    def test_when_segmented_then_timestamp_matches_clock(
        self, segmentation_harness: SegmentationHarness
    ) -> None:
        harness = segmentation_harness
        harness.clock.current = T0 + timedelta(days=1)

        harness.service.run_once()

        (scene_set,) = harness.scene_store.snapshot().values()
        assert scene_set.created_at == T0 + timedelta(days=1)


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
        assert harness.scene_store.list_asset_versions(harness.service.fingerprint) == {VERSION}

    def test_when_listing_fails_then_error_propagates(self) -> None:
        harness = make_segmentation_harness()
        service = harness.variant()
        service._assets = FailingAssetReader()  # the listing is the one thing a pass cannot skip

        with pytest.raises(OSError, match="registry down"):
            service.run_once()
