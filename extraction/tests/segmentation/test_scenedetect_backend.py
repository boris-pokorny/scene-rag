"""The PySceneDetect backends against a synthesized two-scene video."""

from pathlib import Path

import pytest

from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.protocols import VideoAnalysisError

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scene_rag.extraction.segmentation.scenedetect_backend import (  # noqa: E402
    ContentSegmentExtractor,
    SceneDetectKeyframeRenderer,
)

FPS = 25.0
FRAMES_PER_SCENE = 50
SIZE = (320, 240)


@pytest.fixture(scope="module")
def two_scene_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fifty black frames then fifty white ones: a change far above the threshold."""
    path = tmp_path_factory.mktemp("video") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    assert writer.isOpened(), "OpenCV could not open a VideoWriter (MJPG/avi)"
    for color in ((0, 0, 0), (255, 255, 255)):
        frame = np.full((SIZE[1], SIZE[0], 3), color, dtype=np.uint8)
        for _ in range(FRAMES_PER_SCENE):
            writer.write(frame)
    writer.release()
    return path


class TestGivenTwoSceneVideo:
    """The synthesized clip."""

    def test_when_extracted_then_two_segments_tile_the_stream(self, two_scene_video: Path) -> None:
        analysis = ContentSegmentExtractor(SegmentationConfig()).extract(two_scene_video)

        assert analysis.media.frame_count == 2 * FRAMES_PER_SCENE
        assert (analysis.media.width, analysis.media.height) == SIZE
        assert analysis.media.frame_rate == FPS
        assert abs(analysis.media.duration_seconds - 4.0) < 1e-6
        assert [s.index for s in analysis.segments] == [0, 1]
        assert [(s.start_frame, s.end_frame) for s in analysis.segments] == [(0, 50), (50, 100)]
        assert analysis.segments[1].start_seconds == 2.0

    def test_when_threshold_unreachable_then_one_segment(self, two_scene_video: Path) -> None:
        analysis = ContentSegmentExtractor(SegmentationConfig(scene_threshold=255.0)).extract(
            two_scene_video
        )

        assert [(s.start_frame, s.end_frame) for s in analysis.segments] == [(0, 100)]

    def test_when_min_scene_longer_than_first_scene_then_cut_dropped(
        self, two_scene_video: Path
    ) -> None:
        analysis = ContentSegmentExtractor(SegmentationConfig(min_scene_seconds=3.0)).extract(
            two_scene_video
        )

        assert len(analysis.segments) == 1

    def test_when_rendered_then_frames_per_segment_in_order(
        self, two_scene_video: Path, tmp_path: Path
    ) -> None:
        extractor = ContentSegmentExtractor(SegmentationConfig())
        renderer = SceneDetectKeyframeRenderer(SegmentationConfig(keyframe_width=160))
        segments = extractor.extract(two_scene_video).segments

        rendered = renderer.render(two_scene_video, segments, tmp_path / "kf")

        assert set(rendered) == {0, 1}
        for index, frames in rendered.items():
            segment = segments[index]
            assert len(frames) == 2
            assert [f.frame for f in frames] == sorted({f.frame for f in frames})
            assert all(segment.start_frame <= f.frame < segment.end_frame for f in frames)
            for frame in frames:
                assert frame.path.is_file() and frame.path.parent == tmp_path / "kf"
                assert frame.path.name == f"scene_{index + 1:03d}_frame_{frame.frame}.jpg"
                assert frame.seconds == frame.frame / FPS
                image = cv2.imread(str(frame.path))
                assert image.shape[1] == 160

    def test_when_segments_are_a_slice_then_keyed_by_their_index(
        self, two_scene_video: Path, tmp_path: Path
    ) -> None:
        (_, second) = (
            ContentSegmentExtractor(SegmentationConfig()).extract(two_scene_video).segments
        )

        rendered = SceneDetectKeyframeRenderer(SegmentationConfig()).render(
            two_scene_video, [second], tmp_path
        )

        assert set(rendered) == {second.index} == {1}
        assert all(second.start_frame <= f.frame < second.end_frame for f in rendered[1])

    def test_when_segment_shorter_than_picks_then_duplicates_dropped(
        self, two_scene_video: Path, tmp_path: Path
    ) -> None:
        config = SegmentationConfig(keyframes_per_scene=4, keyframe_frame_margin=0)
        extractor = ContentSegmentExtractor(SegmentationConfig())
        (first, _) = extractor.extract(two_scene_video).segments
        tiny = first.__class__(
            index=0, start_seconds=0.0, end_seconds=0.04, start_frame=0, end_frame=1
        )

        rendered = SceneDetectKeyframeRenderer(config).render(two_scene_video, [tiny], tmp_path)

        frames = [f.frame for f in rendered[0]]
        assert frames == sorted(set(frames))
        assert len(frames) < config.keyframes_per_scene


class TestGivenUnreadableFile:
    """Something that is not a video."""

    def test_when_extracted_then_video_analysis_error(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.mp4"
        bogus.write_bytes(b"not a video")

        with pytest.raises(VideoAnalysisError, match="bogus.mp4"):
            ContentSegmentExtractor(SegmentationConfig()).extract(bogus)
        with pytest.raises(VideoAnalysisError, match="bogus.mp4"):
            SceneDetectKeyframeRenderer(SegmentationConfig()).render(bogus, [], tmp_path)
