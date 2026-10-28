from pathlib import Path

import pytest

from scene_rag.extraction.segmentation import fingerprint
from scene_rag.extraction.segmentation.config import SegmentationConfig


class TestGivenPackageSources:
    """A stand-in package directory holding every module of the step."""

    @pytest.fixture
    def package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        for name in (*fingerprint._FINGERPRINTED_MODULES, "bootstrap.py", "scene_store.py"):
            (tmp_path / name).write_text(f"# {name}\n")
        monkeypatch.setattr(fingerprint, "_PACKAGE_DIR", tmp_path)
        return tmp_path

    def test_when_computed_twice_then_same_twelve_hex_digits(self, package: Path) -> None:
        first = fingerprint.compute_fingerprint(SegmentationConfig())

        assert first == fingerprint.compute_fingerprint(SegmentationConfig())
        assert len(first) == 12 and int(first, 16) >= 0

    def test_when_deployment_module_changes_then_unchanged(self, package: Path) -> None:
        before = fingerprint.compute_fingerprint(SegmentationConfig())
        (package / "bootstrap.py").write_text("# new log message\n")
        (package / "scene_store.py").write_text("# new index\n")

        assert fingerprint.compute_fingerprint(SegmentationConfig()) == before

    def test_when_output_module_changes_then_changes(self, package: Path) -> None:
        before = fingerprint.compute_fingerprint(SegmentationConfig())
        (package / "scenedetect_backend.py").write_text("# different detector\n")

        assert fingerprint.compute_fingerprint(SegmentationConfig()) != before

    def test_when_hyperparameter_changes_then_changes(self, package: Path) -> None:
        before = fingerprint.compute_fingerprint(SegmentationConfig())

        assert fingerprint.compute_fingerprint(SegmentationConfig(keyframes_per_scene=3)) != before

    def test_when_listed_module_missing_then_raises(self, package: Path) -> None:
        (package / "service.py").unlink()

        with pytest.raises(FileNotFoundError, match="service.py"):
            fingerprint.compute_fingerprint(SegmentationConfig())


class TestGivenInstalledPackage:
    """The real subpackage on disk."""

    def test_when_computed_then_every_listed_module_exists(self) -> None:
        assert len(fingerprint.compute_fingerprint(SegmentationConfig())) == 12
