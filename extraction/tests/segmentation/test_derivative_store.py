import hashlib
import shutil
import time
from pathlib import Path

import pytest

from scene_rag.extraction.segmentation.derivative_store import FilesystemDerivativeStore
from scene_rag.extraction.segmentation.dispatch import ThreadDispatcher


def write(directory: Path, name: str, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return path


class TestGivenEmptyStore:
    """A store rooted at a directory that does not exist yet."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return tmp_path / "derivatives"

    def test_when_file_put_then_filed_by_content_and_consumed(
        self, root: Path, tmp_path: Path
    ) -> None:
        store = FilesystemDerivativeStore(root, "sha256")
        source = write(tmp_path / "work", "scene_001_frame_12.jpg", b"jpeg-bytes")
        digest = hashlib.sha256(b"jpeg-bytes").hexdigest()

        reference = store.put(source, "image/jpeg")

        filed = root / "sha256" / digest[:2] / f"{digest}.jpg"
        assert reference.digest == f"sha256:{digest}"
        assert reference.uri == filed.resolve().as_uri()
        assert (reference.media_type, reference.size_bytes) == ("image/jpeg", 10)
        assert filed.read_bytes() == b"jpeg-bytes"
        assert not source.exists()
        assert [p.name for p in filed.parent.iterdir()] == [filed.name]

    def test_when_same_bytes_put_twice_then_one_file_same_reference(
        self, root: Path, tmp_path: Path
    ) -> None:
        store = FilesystemDerivativeStore(root, "sha256")
        first = store.put(write(tmp_path / "a", "scene_001.wav", b"wav"), "audio/wav")
        again = write(tmp_path / "b", "scene_002.wav", b"wav")

        second = store.put(again, "audio/wav")

        assert second == first
        assert not again.exists()
        assert len(list(root.rglob("*.wav"))) == 1

    def test_when_same_bytes_put_from_two_threads_at_once_then_one_file_no_error(
        self, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two threads stage the same content at the same moment; the window is held open."""
        real_move = shutil.move

        def slow_move(src: Path, dst: Path) -> None:
            real_move(src, dst)
            time.sleep(0.05)

        monkeypatch.setattr(shutil, "move", slow_move)
        store = FilesystemDerivativeStore(root, "sha256")
        sources = [write(tmp_path / f"w{i}", "scene_001.wav", b"same") for i in range(2)]

        with ThreadDispatcher(2) as threads:
            references = list(threads.map(lambda s: store.put(s, "audio/wav"), sources))

        assert references[0] == references[1]
        assert len(list(root.rglob("*.wav"))) == 1
        assert list(root.rglob("*.tmp")) == []

    def test_when_algorithm_differs_then_filed_under_it(self, root: Path, tmp_path: Path) -> None:
        reference = FilesystemDerivativeStore(root, "sha1").put(
            write(tmp_path / "work", "f.jpg", b"x"), "image/jpeg"
        )

        assert reference.digest == f"sha1:{hashlib.sha1(b'x').hexdigest()}"
        assert "/sha1/" in reference.uri

    def test_when_media_type_unknown_then_rejected_and_file_kept(
        self, root: Path, tmp_path: Path
    ) -> None:
        source = write(tmp_path / "work", "notes.txt", b"text")

        with pytest.raises(ValueError, match="text/plain"):
            FilesystemDerivativeStore(root, "sha256").put(source, "text/plain")

        assert source.exists()

    def test_when_source_missing_then_oserror(self, root: Path, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            FilesystemDerivativeStore(root, "sha256").put(tmp_path / "missing.jpg", "image/jpeg")
