"""Audio clips backed by the ffmpeg command line: the only module running subprocesses.

Clips are 16-bit PCM WAV, a codec-independent format speech models consume directly.
"""

import logging
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.protocols import AudioExtractionError

logger = logging.getLogger(__name__)

# ffmpeg emits this when -map matches no stream, e.g. a video without audio.
_NO_STREAM_MARKER: Final = "matches no streams"
_STDERR_TAIL: Final = 500
# A PCM WAV this small is a header without samples: ffmpeg writes one, and exits 0, for a
# segment lying past the end of an audio stream shorter than the video.
_WAV_HEADER_BYTES: Final = 44


def clip_name(segment: models.Segment) -> str:
    """Return the clip file name of a segment, numbered from one like keyframes."""
    return f"scene_{segment.index + 1:03d}.wav"


class CommandRunner(Protocol):
    """Executes a command line."""

    def run(self, command: Sequence[str]) -> tuple[int, str]:
        """Run the command to completion.

        Returns:
            Its exit code and standard error output.
        """
        ...


class SubprocessRunner:
    """`CommandRunner` over `subprocess.run`."""

    def run(self, command: Sequence[str]) -> tuple[int, str]:
        """Run the command, capturing its output."""
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace", check=False
        )
        return result.returncode, result.stderr


class FfmpegAudioClipper:
    """`AudioClipper` slicing one WAV per segment out of a media file."""

    def __init__(self, config: SegmentationConfig, runner: CommandRunner, binary: str) -> None:
        """Take sample rate and channels from ``config``; run ``binary`` through ``runner``."""
        self._sample_rate = config.audio_sample_rate
        self._channels = config.audio_channels
        self._runner = runner
        self._binary = binary

    def clip(
        self, path: Path, segments: Sequence[models.Segment], output_dir: Path
    ) -> Mapping[int, Path]:
        """Write one clip per segment that has samples; see `AudioClipper.clip`."""
        if not self._has_audio_stream(path):
            logger.info("no audio stream in %s", path.name)
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        clips: dict[int, Path] = {}
        for segment in segments:
            if segment.duration_seconds <= 0:
                continue
            destination = output_dir / clip_name(segment)
            self._extract(path, segment, destination)
            if destination.stat().st_size <= _WAV_HEADER_BYTES:
                logger.info("no samples for segment %d of %s", segment.index, path.name)
                destination.unlink()
                continue
            clips[segment.index] = destination
        logger.info("clipped %d segment(s) of %s", len(clips), path.name)
        return clips

    def _has_audio_stream(self, path: Path) -> bool:
        """Decode one audio frame to nothing; cheaper than a full probe."""
        code, stderr = self._runner.run(
            [
                *self._base_command(),
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-frames:a",
                "1",
                "-f",
                "null",
                "-",
            ]
        )
        if code == 0:
            return True
        if _NO_STREAM_MARKER in stderr:
            return False
        raise AudioExtractionError(
            f"ffmpeg failed probing {path}: {stderr.strip()[-_STDERR_TAIL:]}"
        )

    def _extract(self, path: Path, segment: models.Segment, destination: Path) -> None:
        code, stderr = self._runner.run(
            [
                *self._base_command(),
                "-y",
                # Input seeking: fast, and sample-accurate once decoding to PCM.
                "-ss",
                f"{segment.start_seconds:.6f}",
                "-t",
                f"{segment.duration_seconds:.6f}",
                "-i",
                str(path),
                "-vn",
                "-map",
                "0:a:0",
                "-acodec",
                "pcm_s16le",
                "-ac",
                str(self._channels),
                "-ar",
                str(self._sample_rate),
                str(destination),
            ]
        )
        if code != 0:
            raise AudioExtractionError(
                f"ffmpeg failed extracting {destination.name} from {path}: "
                f"{stderr.strip()[-_STDERR_TAIL:]}"
            )

    def _base_command(self) -> list[str]:
        return [self._binary, "-hide_banner", "-nostdin", "-v", "error"]
