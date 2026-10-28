"""`FfmpegAudioClipper` command construction without ffmpeg, and one case with it."""

import shutil
import subprocess
import threading
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.extraction.segmentation.ffmpeg_backend import (
    FfmpegAudioClipper,
    SubprocessRunner,
    clip_name,
)
from scene_rag.extraction.segmentation.protocols import AudioExtractionError, Dispatcher
from tests.support.segmentation_fakes import ConcurrencyGauge, make_segments

PREFIX = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
VIDEO = Path("/videos/clip.mp4")


# Bytes a fake extraction writes: a header plus one sample, or a header alone.
SAMPLES = b"\0" * 46
HEADER_ONLY = b"\0" * 44


class FakeRunner:
    """CommandRunner returning canned results per invocation; records commands.

    An extraction command (one with an output file, the last argument) writes ``content``
    there when it succeeds, as ffmpeg would.
    """

    def __init__(
        self,
        results: Sequence[tuple[int, str]] = (),
        content: bytes = SAMPLES,
        meet: threading.Barrier | None = None,
    ) -> None:
        self._results = list(results)
        self._content = content
        self._meet = meet
        self._lock = threading.Lock()
        self.gauge = ConcurrencyGauge()
        self.commands: list[list[str]] = []

    def run(self, command: Sequence[str]) -> tuple[int, str]:
        with self._lock:
            self.commands.append(list(command))
            code, stderr = self._results.pop(0) if self._results else (0, "")
        extraction = command[-1].endswith(".wav")
        self.gauge.enter()
        try:
            if extraction and self._meet is not None:
                self._meet.wait()
        finally:
            self.gauge.leave()
        if code == 0 and extraction:
            Path(command[-1]).write_bytes(self._content)
        return code, stderr


def clipper(
    runner: FakeRunner,
    config: SegmentationConfig | None = None,
    dispatcher: Dispatcher | None = None,
) -> FfmpegAudioClipper:
    return FfmpegAudioClipper(
        config or SegmentationConfig(), runner, "ffmpeg", dispatcher or SerialDispatcher()
    )


class TestGivenFakeRunner:
    """Commands the clipper issues, checked without ffmpeg."""

    def test_when_clipped_then_probe_then_one_command_per_segment(self, tmp_path: Path) -> None:
        runner = FakeRunner()

        clips = clipper(runner).clip(VIDEO, make_segments(((0.0, 4.0), (4.0, 8.5))), tmp_path)

        assert clips == {0: tmp_path / "scene_001.wav", 1: tmp_path / "scene_002.wav"}
        assert runner.commands[0] == [
            *PREFIX, "-i", str(VIDEO), "-map", "0:a:0", "-frames:a", "1", "-f", "null", "-",
        ]  # fmt: skip
        assert runner.commands[1] == [
            *PREFIX, "-y", "-ss", "0.000000", "-t", "4.000000", "-i", str(VIDEO), "-vn",
            "-map", "0:a:0", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            str(tmp_path / "scene_001.wav"),
        ]  # fmt: skip
        assert runner.commands[2][runner.commands[2].index("-ss") + 1] == "4.000000"
        assert runner.commands[2][runner.commands[2].index("-t") + 1] == "4.500000"

    def test_when_config_tuned_then_rate_and_channels_follow(self, tmp_path: Path) -> None:
        runner = FakeRunner()

        clipper(runner, SegmentationConfig(audio_sample_rate=44100, audio_channels=2)).clip(
            VIDEO, make_segments(((0.0, 1.0),)), tmp_path
        )

        command = runner.commands[1]
        assert command[command.index("-ar") + 1] == "44100"
        assert command[command.index("-ac") + 1] == "2"

    def test_when_no_audio_stream_then_no_clips_and_no_extraction(self, tmp_path: Path) -> None:
        runner = FakeRunner([(1, "Stream map '0:a:0' matches no streams.")])

        clips = clipper(runner).clip(VIDEO, make_segments(), tmp_path)

        assert clips == {}
        assert len(runner.commands) == 1

    def test_when_probe_fails_then_raises(self, tmp_path: Path) -> None:
        runner = FakeRunner([(1, "clip.mp4: Invalid data found when processing input")])

        with pytest.raises(AudioExtractionError, match="Invalid data"):
            clipper(runner).clip(VIDEO, make_segments(), tmp_path)

    def test_when_extraction_fails_then_raises_naming_clip(self, tmp_path: Path) -> None:
        runner = FakeRunner([(0, ""), (1, "Error while decoding stream")])

        with pytest.raises(AudioExtractionError, match="scene_001.wav"):
            clipper(runner).clip(VIDEO, make_segments(), tmp_path)

    def test_when_segment_empty_then_skipped(self, tmp_path: Path) -> None:
        runner = FakeRunner()

        clips = clipper(runner).clip(VIDEO, make_segments(((2.0, 2.0), (2.0, 5.0))), tmp_path)

        assert clips == {1: tmp_path / "scene_002.wav"}
        assert len(runner.commands) == 2

    def test_when_audio_ends_before_segment_then_no_clip_for_it(self, tmp_path: Path) -> None:
        runner = FakeRunner(content=HEADER_ONLY)

        clips = clipper(runner).clip(VIDEO, make_segments(((0.0, 4.0),)), tmp_path)

        assert clips == {}
        assert not (tmp_path / "scene_001.wav").exists()
        assert len(runner.commands) == 2

    def test_when_named_then_numbered_from_one(self) -> None:
        assert [clip_name(s) for s in make_segments()] == [
            "scene_001.wav",
            "scene_002.wav",
            "scene_003.wav",
        ]


class TestGivenThreadDispatcher:
    """The same clips, several ffmpeg processes at a time."""

    def test_when_clipped_then_segments_extracted_at_once_and_mapping_unchanged(
        self, tmp_path: Path
    ) -> None:
        serial = FakeRunner()
        clipper(serial).clip(VIDEO, make_segments(), tmp_path / "serial")
        parallel = FakeRunner(meet=threading.Barrier(3, timeout=5))

        with ThreadDispatcher(3) as threads:
            clips = clipper(parallel, dispatcher=threads).clip(
                VIDEO, make_segments(), tmp_path / "parallel"
            )

        assert clips == {i: tmp_path / "parallel" / f"scene_00{i + 1}.wav" for i in range(3)}
        assert parallel.gauge.peak == 3
        assert parallel.commands[0] == serial.commands[0]  # the probe, first and alone
        relocate = str(tmp_path / "serial"), str(tmp_path / "parallel")
        assert sorted(parallel.commands[1:]) == sorted(
            [arg.replace(*relocate) for arg in command] for command in serial.commands[1:]
        )

    def test_when_one_extraction_fails_then_raises_naming_its_clip(self, tmp_path: Path) -> None:
        runner = FakeRunner([(0, ""), (0, ""), (1, "Error while decoding stream"), (0, "")])

        with (
            ThreadDispatcher(2) as threads,
            pytest.raises(AudioExtractionError, match="scene_002.wav"),
        ):
            clipper(runner, dispatcher=threads).clip(VIDEO, make_segments(), tmp_path)


@pytest.mark.integration
class TestGivenFfmpeg:
    """The real ffmpeg on a synthesized tone."""

    @pytest.fixture
    def tone(self, tmp_path: Path) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("no ffmpeg on PATH")
        path = tmp_path / "tone.mp4"
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3", "-shortest", str(path)],
            check=True,
        )  # fmt: skip
        return path

    def test_when_clipped_then_wav_of_segment_length(self, tone: Path, tmp_path: Path) -> None:
        clips = FfmpegAudioClipper(
            SegmentationConfig(), SubprocessRunner(), "ffmpeg", SerialDispatcher()
        ).clip(tone, make_segments(((0.0, 1.0), (1.0, 2.5))), tmp_path / "audio")

        with wave.open(str(clips[1])) as clip:
            assert clip.getnchannels() == 1
            assert clip.getframerate() == 16000
            assert abs(clip.getnframes() / 16000 - 1.5) < 0.05

    def test_when_clipped_on_threads_then_same_bytes_as_serial(
        self, tone: Path, tmp_path: Path
    ) -> None:
        segments = make_segments(((0.0, 0.5), (0.5, 1.5), (1.5, 2.0), (2.0, 3.0)))
        serial = FfmpegAudioClipper(
            SegmentationConfig(), SubprocessRunner(), "ffmpeg", SerialDispatcher()
        ).clip(tone, segments, tmp_path / "serial")

        with ThreadDispatcher(4) as threads:
            parallel = FfmpegAudioClipper(
                SegmentationConfig(), SubprocessRunner(), "ffmpeg", threads
            ).clip(tone, segments, tmp_path / "parallel")

        assert parallel.keys() == serial.keys() == {0, 1, 2, 3}
        for index, clip in serial.items():
            assert parallel[index].read_bytes() == clip.read_bytes()
