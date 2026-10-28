"""The step fingerprint: hyperparameters and output-defining code."""

from pathlib import Path

import pytest

from scene_rag.retrieval.chunking import fingerprint, per_scene
from scene_rag.retrieval.chunking.config import ChunkingConfig, PerSceneConfig, UpstreamConfig

# Modules of the step whose changes must never imply reprocessing.
DEPLOYMENT_MODULES = (
    "bootstrap.py",
    "config.py",
    "settings.py",
    "chunk_store.py",
    "scene_api_client.py",
    "http_transport.py",
    "scheduler.py",
    "service.py",
    "dispatch.py",
    "protocols.py",
)


def compute(config: ChunkingConfig | None = None) -> str:
    return fingerprint.compute_fingerprint(
        ChunkingConfig() if config is None else config, per_scene.FINGERPRINTED_MODULES
    )


class TestGivenPackageSources:
    """A stand-in package directory holding every module of the step."""

    @pytest.fixture
    def package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        for name in (*per_scene.FINGERPRINTED_MODULES, *DEPLOYMENT_MODULES):
            (tmp_path / name).write_text(f"# {name}\n")
        monkeypatch.setattr(fingerprint, "_PACKAGE_DIR", tmp_path)
        return tmp_path

    def test_when_computed_twice_then_same_twelve_hex_digits(self, package: Path) -> None:
        first = compute()

        assert first == compute()
        assert len(first) == 12 and int(first, 16) >= 0

    def test_when_deployment_module_changes_then_unchanged(self, package: Path) -> None:
        before = compute()
        for name in DEPLOYMENT_MODULES:
            (package / name).write_text(f"# {name}: new log message, index or worker count\n")

        assert compute() == before

    @pytest.mark.parametrize("name", per_scene.FINGERPRINTED_MODULES)
    def test_when_output_module_changes_then_changes(self, package: Path, name: str) -> None:
        before = compute()
        (package / name).write_text("# different output\n")

        assert compute() != before

    @pytest.mark.parametrize(
        "config",
        [
            ChunkingConfig(per_scene=PerSceneConfig(section_separator="\n")),
            ChunkingConfig(upstream=UpstreamConfig(enrichment_pins=("description:0123456789ab",))),
            ChunkingConfig(upstream=UpstreamConfig(segmentation_fingerprint="3b9e1f6a2c40")),
            ChunkingConfig(hash_algorithm="sha512"),
        ],
    )
    def test_when_any_hyperparameter_changes_then_changes(
        self, package: Path, config: ChunkingConfig
    ) -> None:
        assert compute(config) != compute()

    def test_when_listed_module_missing_then_raises(self, package: Path) -> None:
        (package / "per_scene.py").unlink()

        with pytest.raises(FileNotFoundError, match="per_scene.py"):
            compute()


class TestGivenInstalledPackage:
    """The real subpackage on disk."""

    def test_when_computed_then_every_listed_module_exists(self) -> None:
        assert len(compute()) == 12
