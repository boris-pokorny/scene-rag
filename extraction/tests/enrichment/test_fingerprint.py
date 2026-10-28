"""Per-enricher fingerprints: hyperparameters, output-defining code and model digest."""

from pathlib import Path

import pytest

from scene_rag.extraction.enrichment import description, fingerprint, transcription
from scene_rag.extraction.enrichment.config import (
    DescriptionConfig,
    EnrichmentConfig,
    TranscriptionConfig,
)

# Modules of the step whose changes must never imply reprocessing.
DEPLOYMENT_MODULES = (
    "bootstrap.py",
    "config.py",
    "enrichment_store.py",
    "scene_store.py",
    "asset_registry.py",
    "scheduler.py",
    "service.py",
    "dispatch.py",
)
DIGEST = "sha256:" + "ab" * 32
DESCRIPTION = DescriptionConfig()
TRANSCRIPTION = TranscriptionConfig()


def describe(config: DescriptionConfig = DESCRIPTION, digest: str = DIGEST) -> str:
    return fingerprint.compute_fingerprint(config, description.FINGERPRINTED_MODULES, digest)


def transcribe(config: TranscriptionConfig = TRANSCRIPTION, digest: str = DIGEST) -> str:
    return fingerprint.compute_fingerprint(config, transcription.FINGERPRINTED_MODULES, digest)


class TestGivenPackageSources:
    """A stand-in package directory holding every module of the step."""

    @pytest.fixture
    def package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        for name in (
            *description.FINGERPRINTED_MODULES,
            *transcription.FINGERPRINTED_MODULES,
            *DEPLOYMENT_MODULES,
        ):
            (tmp_path / name).write_text(f"# {name}\n")
        monkeypatch.setattr(fingerprint, "_PACKAGE_DIR", tmp_path)
        return tmp_path

    def test_when_computed_twice_then_same_twelve_hex_digits(self, package: Path) -> None:
        first = describe()

        assert first == describe()
        assert len(first) == 12 and int(first, 16) >= 0

    def test_when_deployment_module_changes_then_unchanged(self, package: Path) -> None:
        before = (describe(), transcribe())
        for name in DEPLOYMENT_MODULES:
            (package / name).write_text(f"# {name}: new log message, index or worker count\n")

        assert (describe(), transcribe()) == before

    def test_when_own_module_changes_then_only_that_enricher_changes(self, package: Path) -> None:
        before = (describe(), transcribe())
        (package / "ollama_backend.py").write_text("# different request shape\n")

        assert describe() != before[0]
        assert transcribe() == before[1]

    def test_when_shared_module_changes_then_both_change(self, package: Path) -> None:
        before = (describe(), transcribe())
        (package / "models.py").write_text("# new field\n")

        assert describe() != before[0]
        assert transcribe() != before[1]

    def test_when_own_hyperparameter_changes_then_only_that_enricher_changes(
        self, package: Path
    ) -> None:
        config = EnrichmentConfig(transcription=TranscriptionConfig(beam_size=1))

        assert describe(config.description) == describe()
        assert transcribe(config.transcription) != transcribe()

    def test_when_model_digest_changes_then_changes(self, package: Path) -> None:
        assert describe(digest="sha256:" + "cd" * 32) != describe()

    def test_when_listed_module_missing_then_raises(self, package: Path) -> None:
        (package / "description.py").unlink()

        with pytest.raises(FileNotFoundError, match="description.py"):
            describe()


class TestGivenInstalledPackage:
    """The real subpackage on disk."""

    def test_when_computed_then_every_listed_module_exists(self) -> None:
        assert len(describe()) == 12
        assert len(transcribe()) == 12
