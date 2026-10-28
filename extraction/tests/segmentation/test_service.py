"""The per-version unit of work: one live version in, one scene set out."""

from datetime import timedelta
from pathlib import Path

import pytest

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.protocols import (
    Analysis,
    MediaUnavailableError,
    VersionRetiredError,
)
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
    make_view,
    sha256_version,
)

VERSION = sha256_version(b"hello")


class TestGivenLiveVersion:
    """One live version registered, nothing segmented yet."""

    @pytest.fixture
    def segmentation_harness(
        self, segmentation_harness: SegmentationHarness
    ) -> SegmentationHarness:
        segmentation_harness.assets.records[VERSION] = make_view(VERSION)
        return segmentation_harness

    @pytest.fixture
    def record(self, segmentation_harness: SegmentationHarness) -> models.AssetVersionView:
        return segmentation_harness.assets.records[VERSION]

    def test_when_segmented_then_scene_set_written(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness

        outcome = harness.service.segment(record)

        assert outcome == "segmented"
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

    def test_when_segmented_then_keyframes_and_clips_filed_by_content(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness

        harness.service.segment(record)

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
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness

        harness.variant_service(keyframes=FakeKeyframeRenderer(duplicate_frames=True)).segment(
            record
        )

        (scene_set,) = harness.scene_store.snapshot().values()
        assert [k.frame for k in scene_set.scenes[0].keyframes] == [0, 0]
        assert len({k.derivative.digest for k in scene_set.scenes[0].keyframes}) == 1
        assert len(harness.derivatives.blobs) == 3 + 3

    def test_when_no_audio_stream_then_clips_absent(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness

        outcome = harness.variant_service(audio=FakeAudioClipper(silent=True)).segment(record)

        assert outcome == "segmented"
        (scene_set,) = harness.scene_store.snapshot().values()
        assert all(scene.audio_clip is None for scene in scene_set.scenes)

    def test_when_video_has_no_cuts_then_one_scene(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness
        whole: Analysis = make_analysis(bounds=((0.0, 12.5),))

        harness.variant_service(extractor=FakeSegmentExtractor(whole)).segment(record)

        (scene_set,) = harness.scene_store.snapshot().values()
        assert len(scene_set.scenes) == 1
        assert (scene_set.scenes[0].start_frame, scene_set.scenes[0].end_frame) == (0, 312)

    def test_when_source_unreadable_then_media_unavailable_and_nothing_written(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness
        service = harness.variant_service(locator=FakeLocator(available=False))

        with pytest.raises(MediaUnavailableError, match=VERSION):
            service.segment(record)

        assert harness.writer.commands == []
        assert harness.derivatives.puts == []

    def test_when_retired_meanwhile_then_version_retired_before_write(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness

        class RetiringExtractor(FakeSegmentExtractor):
            def extract(self, path: Path) -> Analysis:
                harness.assets.retire(VERSION, at=T0 + timedelta(seconds=1))
                return super().extract(path)

        service = harness.variant_service(extractor=RetiringExtractor())

        with pytest.raises(VersionRetiredError, match=VERSION):
            service.segment(record)

        assert harness.writer.commands == []
        assert harness.scene_store.snapshot() == {}

    def test_when_another_worker_wrote_first_then_duplicate(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness

        class RacingExtractor(FakeSegmentExtractor):
            def extract(self, path: Path) -> Analysis:
                harness.variant_service(extractor=FakeSegmentExtractor()).segment(record)
                return super().extract(path)

        outcome = harness.variant_service(extractor=RacingExtractor()).segment(record)

        assert outcome == "duplicate"
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
    def test_when_collaborator_fails_then_error_propagates_and_nothing_written(
        self,
        segmentation_harness: SegmentationHarness,
        record: models.AssetVersionView,
        broken: dict[str, object],
    ) -> None:
        harness = segmentation_harness
        service = harness.variant_service(**broken)  # type: ignore[arg-type]

        with pytest.raises(Exception, match="could not open|ffmpeg failed|store full"):
            service.segment(record)

        assert harness.writer.commands == []

    def test_when_writer_fails_then_error_propagates_after_the_command(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness
        failing = RecordingSceneWriter(harness.scene_store, fail=True)
        service = harness.variant_service(writer=failing)

        with pytest.raises(RuntimeError, match="injected failure"):
            service.segment(record)

        assert failing.commands == [("put", VERSION)]
        assert harness.scene_store.snapshot() == {}

    def test_when_workspace_root_given_then_scratch_files_live_under_it_and_are_removed(
        self,
        segmentation_harness: SegmentationHarness,
        record: models.AssetVersionView,
        tmp_path: Path,
    ) -> None:
        harness = segmentation_harness
        service = harness.variant_service(workspace_root=tmp_path)

        service.segment(record)

        assert harness.derivatives.paths
        assert all(path.is_relative_to(tmp_path) for path in harness.derivatives.paths)
        assert list(tmp_path.iterdir()) == []

    def test_when_segmented_then_timestamp_matches_clock(
        self, segmentation_harness: SegmentationHarness, record: models.AssetVersionView
    ) -> None:
        harness = segmentation_harness
        harness.clock.current = T0 + timedelta(days=1)

        harness.service.segment(record)

        (scene_set,) = harness.scene_store.snapshot().values()
        assert scene_set.created_at == T0 + timedelta(days=1)
