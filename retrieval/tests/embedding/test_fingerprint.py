"""The step fingerprint: hyperparameters, output-defining code and the model digest."""

from pathlib import Path

import pytest

from scene_rag.retrieval.embedding import fingerprint, ollama_embedding
from scene_rag.retrieval.embedding.config import (
    EmbeddingConfig,
    IndexConfig,
    ModelConfig,
    UpstreamConfig,
)

# Modules of the step whose changes must never imply re-embedding.
DEPLOYMENT_MODULES = (
    "bootstrap.py",
    "config.py",
    "settings.py",
    "chunk_reader.py",
    "index_store.py",
    "index_setup.py",
    "qdrant_index.py",
    "http_transport.py",
    "ollama_pull.py",
    "scheduler.py",
    "service.py",
    "dispatch.py",
    "report.py",
    "protocols.py",
)
DIGEST = "sha256:" + "ab" * 32


def compute(config: EmbeddingConfig | None = None, digest: str = DIGEST) -> str:
    return fingerprint.compute_fingerprint(
        EmbeddingConfig() if config is None else config,
        ollama_embedding.FINGERPRINTED_MODULES,
        digest,
    )


class TestGivenPackageSources:
    """A stand-in package directory holding every module of the step."""

    @pytest.fixture
    def package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        for name in (*ollama_embedding.FINGERPRINTED_MODULES, *DEPLOYMENT_MODULES):
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
            (package / name).write_text(f"# {name}: new log message, timeout or batch size\n")

        assert compute() == before

    @pytest.mark.parametrize("name", ollama_embedding.FINGERPRINTED_MODULES)
    def test_when_output_module_changes_then_changes(self, package: Path, name: str) -> None:
        before = compute()
        (package / name).write_text("# different output\n")

        assert compute() != before

    def test_when_model_digest_changes_then_changes(self, package: Path) -> None:
        assert compute(digest="sha256:" + "cd" * 32) != compute()

    @pytest.mark.parametrize(
        "config",
        [
            EmbeddingConfig(upstream=UpstreamConfig(chunking_fingerprint="0123456789ab")),
            EmbeddingConfig(model=ModelConfig(name="nomic-embed-text")),
            EmbeddingConfig(model=ModelConfig(document_prefix="")),
            EmbeddingConfig(model=ModelConfig(query_prefix="query: ")),
            EmbeddingConfig(model=ModelConfig(truncate=False)),
            EmbeddingConfig(model=ModelConfig(num_ctx=512)),
            EmbeddingConfig(index=IndexConfig(distance="dot")),
            EmbeddingConfig(hash_algorithm="sha512"),
        ],
    )
    def test_when_any_hyperparameter_changes_then_changes(
        self, package: Path, config: EmbeddingConfig
    ) -> None:
        assert compute(config) != compute()

    def test_when_listed_module_missing_then_raises(self, package: Path) -> None:
        (package / "ollama_embedding.py").unlink()

        with pytest.raises(FileNotFoundError, match="ollama_embedding.py"):
            compute()


class TestGivenInstalledPackage:
    """The real subpackage on disk."""

    def test_when_computed_then_every_listed_module_exists(self) -> None:
        assert len(compute()) == 12
