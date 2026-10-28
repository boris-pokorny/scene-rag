"""Scene cuts and keyframes backed by PySceneDetect: the only module importing it or OpenCV.

`ContentSegmentExtractor` places cuts where the content detector sees the frame change above
the threshold; `SceneDetectKeyframeRenderer` renders evenly spaced, resized JPEGs per segment.
OpenCV's decoder log level is an environment variable; `bootstrap` sets it before importing
this module, as only bootstrap touches the environment.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from scenedetect import (
    ContentDetector,
    FrameTimecode,
    SceneManager,
    VideoStream,
    open_video,
    save_images,
)

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.protocols import (
    Analysis,
    RenderedFrame,
    VideoAnalysisError,
)

logger = logging.getLogger(__name__)

_IMAGE_NAME_TEMPLATE: Final = "scene_${SCENE_NUMBER}_frame_${FRAME_NUMBER}"
_IMAGE_NAME: Final = re.compile(r"scene_(\d+)_frame_(\d+)\.jpg$")


def _open(path: Path) -> VideoStream:
    try:
        return open_video(str(path))
    except Exception as exc:
        raise VideoAnalysisError(f"could not open video {path}: {exc}") from exc


class ContentSegmentExtractor:
    """`SegmentExtractor` cutting at content changes; a video without cuts is one segment."""

    def __init__(self, config: SegmentationConfig) -> None:
        """Take the threshold and minimum scene length from ``config``."""
        self._threshold = config.scene_threshold
        self._min_scene_seconds = config.min_scene_seconds

    def extract(self, path: Path) -> Analysis:
        """Detect the scenes of the video at ``path``; see `SegmentExtractor.extract`."""
        video = _open(path)
        duration = video.duration
        if duration is None or duration.frame_num == 0:
            raise VideoAnalysisError(f"video has no frames: {path}")
        frame_rate = float(video.frame_rate)
        media = models.MediaInfo(
            duration_seconds=duration.seconds,
            frame_rate=frame_rate,
            width=video.frame_size[0],
            height=video.frame_size[1],
            frame_count=duration.frame_num,
        )
        manager = SceneManager()
        manager.add_detector(
            ContentDetector(
                threshold=self._threshold,
                min_scene_len=round(frame_rate * self._min_scene_seconds),
            )
        )
        manager.detect_scenes(video=video)
        pairs = manager.get_scene_list() or [(FrameTimecode(0, video.frame_rate), duration)]
        segments = tuple(
            models.Segment(
                index=index,
                start_seconds=start.seconds,
                end_seconds=end.seconds,
                start_frame=start.frame_num,
                end_frame=end.frame_num,
            )
            for index, (start, end) in enumerate(pairs)
        )
        logger.info(
            "found %d segment(s) in %s (%d frames)", len(segments), path.name, media.frame_count
        )
        return Analysis(media=media, segments=segments)


class SceneDetectKeyframeRenderer:
    """`KeyframeRenderer` over PySceneDetect's ``save_images``."""

    def __init__(self, config: SegmentationConfig) -> None:
        """Take the count, size, margin and quality of keyframes from ``config``."""
        self._num_images = config.keyframes_per_scene
        self._width = config.keyframe_width
        self._frame_margin = config.keyframe_frame_margin
        self._quality = config.keyframe_quality

    def render(
        self, path: Path, segments: Sequence[models.Segment], output_dir: Path
    ) -> Mapping[int, Sequence[RenderedFrame]]:
        """Render keyframes of every segment; see `KeyframeRenderer.render`.

        Short segments clamp every pick to the same frame; those duplicates are dropped.
        """
        video = _open(path)
        frame_rate = video.frame_rate
        pairs = [
            (
                FrameTimecode(segment.start_frame, frame_rate),
                FrameTimecode(segment.end_frame, frame_rate),
            )
            for segment in segments
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = save_images(
            pairs,
            video,
            num_images=self._num_images,
            frame_margin=self._frame_margin,
            width=self._width,
            encoder_param=self._quality,
            image_name_template=_IMAGE_NAME_TEMPLATE,
            output_dir=output_dir,
        )
        # The returned keys count scenes from one in some releases and from zero in others;
        # the file names, from the template, always count from one, by position in `pairs`.
        frames: dict[int, dict[int, RenderedFrame]] = {}
        for names in saved.values():
            for name in names:
                match = _IMAGE_NAME.search(name)
                if match is None:
                    raise VideoAnalysisError(f"unexpected keyframe name {name!r}")
                index, frame = segments[int(match.group(1)) - 1].index, int(match.group(2))
                file = Path(name) if Path(name).is_absolute() else output_dir / Path(name).name
                frames.setdefault(index, {}).setdefault(
                    frame, RenderedFrame(path=file, frame=frame, seconds=frame / float(frame_rate))
                )
        rendered = {
            index: [by_frame[frame] for frame in sorted(by_frame)]
            for index, by_frame in sorted(frames.items())
        }
        logger.info(
            "rendered %d keyframe(s) for %s", sum(len(f) for f in rendered.values()), path.name
        )
        return rendered
