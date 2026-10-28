"""Loading the faster-whisper model: the snapshot digest, the patched library, the real one."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scene_rag.extraction.enrichment import whisper_loader
from scene_rag.extraction.enrichment.config import TranscriptionConfig
from scene_rag.extraction.enrichment.models import ModelRef
from scene_rag.extraction.enrichment.whisper_loader import snapshot_digest
from tests.enrichment.test_whisper_backend import FakeWhisperModel


class TestGivenSnapshotDirectory:
    """A downloaded model directory."""

    def test_when_digested_then_only_model_files_count(self, tmp_path: Path) -> None:
        (tmp_path / "model.bin").write_bytes(b"weights")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "README.md").write_text("docs")
        before = snapshot_digest(tmp_path)

        (tmp_path / "README.md").write_text("other docs")
        assert snapshot_digest(tmp_path) == before
        (tmp_path / "model.bin").write_bytes(b"other weights")
        assert snapshot_digest(tmp_path) != before
        assert before.startswith("sha256:")

    @pytest.mark.parametrize(
        "name", ["preprocessor_config.json", "tokenizer.json", "vocabulary.json", "vocabulary.txt"]
    )
    def test_when_tokenizer_or_feature_file_changes_then_digest_changes(
        self, tmp_path: Path, name: str
    ) -> None:
        (tmp_path / "model.bin").write_bytes(b"weights")
        before = snapshot_digest(tmp_path)

        (tmp_path / name).write_text("{}")

        assert snapshot_digest(tmp_path) != before

    def test_when_no_model_files_then_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="model.bin"):
            snapshot_digest(tmp_path)


class TestGivenLoader:
    """`load_transcriber` over a patched library."""

    def test_when_library_missing_then_runtime_error_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "faster_whisper", None)

        with pytest.raises(RuntimeError, match="pdm install -G enrichment"):
            whisper_loader.load_transcriber(TranscriptionConfig(), cache_dir=None, cpu_threads=0)

    def test_when_download_fails_then_runtime_error_names_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import faster_whisper

        def download_model(name: str, **options: object) -> str:
            raise OSError("offline")

        monkeypatch.setattr(faster_whisper, "download_model", download_model)

        with pytest.raises(RuntimeError, match="speech model 'small'.*offline"):
            whisper_loader.load_transcriber(TranscriptionConfig(), cache_dir=None, cpu_threads=0)

    def test_when_snapshot_present_then_model_built_from_it_with_digest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import faster_whisper

        (tmp_path / "model.bin").write_bytes(b"weights")
        built: list[tuple[str, dict[str, object]]] = []

        def download_model(name: str, **options: object) -> str:
            assert options == {"cache_dir": str(tmp_path)}
            return str(tmp_path)

        class WhisperModel(FakeWhisperModel):
            def __init__(self, path: str, **options: object) -> None:
                super().__init__([])
                built.append((path, options))

        monkeypatch.setattr(faster_whisper, "download_model", download_model)
        monkeypatch.setattr(faster_whisper, "WhisperModel", WhisperModel)

        transcriber = whisper_loader.load_transcriber(
            TranscriptionConfig(compute_type="float32"), cache_dir=tmp_path, cpu_threads=2
        )

        assert transcriber.model == ModelRef("small", snapshot_digest(tmp_path))
        assert built == [
            (str(tmp_path), {"device": "cpu", "compute_type": "float32", "cpu_threads": 2})
        ]


@pytest.mark.integration
@pytest.mark.models
class TestGivenRealTinyModel:
    """The real library and the ``tiny`` model; opt in with SCENE_RAG_EXTRACTION_TEST_MODELS=1."""

    def test_when_tone_transcribed_then_no_crash_and_digest_stable(self, tmp_path: Path) -> None:
        if os.environ.get("SCENE_RAG_EXTRACTION_TEST_MODELS") != "1":
            pytest.skip("set SCENE_RAG_EXTRACTION_TEST_MODELS=1 to download and run the tiny model")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("no ffmpeg on PATH")
        clip = tmp_path / "tone.wav"
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-ac", "1", "-ar", "16000", str(clip)],
            check=True,
        )  # fmt: skip

        first = whisper_loader.load_transcriber(
            TranscriptionConfig(model="tiny"), cache_dir=None, cpu_threads=0
        )
        again = whisper_loader.load_transcriber(
            TranscriptionConfig(model="tiny"), cache_dir=None, cpu_threads=0
        )
        transcript = first.transcribe(clip.read_bytes())

        assert first.model == again.model and first.model.digest.startswith("sha256:")
        assert all(s.end_seconds >= s.start_seconds for s in transcript.segments)
