"""The query_config id: hyperparameters, output-defining code and the generator digest."""

from pathlib import Path

import pytest

from scene_rag.retrieval.query_api import fingerprint
from scene_rag.retrieval.query_api.config import (
    GeneratorConfig,
    PromptConfig,
    QueryConfig,
    RefusalConfig,
    RetrieverConfig,
    UpstreamConfig,
)

# Modules of the role whose changes must never imply a new release.
DEPLOYMENT_MODULES = (
    "bootstrap.py",
    "config.py",
    "settings.py",
    "protocols.py",
    "service.py",
    "index_ledger.py",
    "search_index.py",
    "scene_api_client.py",
    "http_transport.py",
    "ollama_pull.py",
    "caller.py",
    "encoding.py",
    "app.py",
    "mcp_server.py",
    "openapi.py",
    "mcp_schema.py",
)
DIGEST = "sha256:" + "ab" * 32


def compute(config: QueryConfig | None = None, digest: str = DIGEST) -> str:
    return fingerprint.compute_query_config_id(
        QueryConfig() if config is None else config, fingerprint.FINGERPRINTED_MODULES, digest
    )


class TestGivenPackageSources:
    """A stand-in package directory holding every module of the role."""

    @pytest.fixture
    def package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        for name in (*fingerprint.FINGERPRINTED_MODULES, *DEPLOYMENT_MODULES):
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
            (package / name).write_text(f"# {name}: new log message or timeout\n")

        assert compute() == before

    @pytest.mark.parametrize("name", fingerprint.FINGERPRINTED_MODULES)
    def test_when_output_module_changes_then_changes(self, package: Path, name: str) -> None:
        before = compute()
        (package / name).write_text("# different output\n")

        assert compute() != before

    def test_when_generator_digest_changes_then_changes(self, package: Path) -> None:
        assert compute(digest="sha256:" + "cd" * 32) != compute()

    @pytest.mark.parametrize(
        "config",
        [
            QueryConfig(upstream=UpstreamConfig(embedding_fingerprint="0123456789ab")),
            QueryConfig(retriever=RetrieverConfig(top_k=3)),
            QueryConfig(refusal=RefusalConfig(min_score=0.5)),
            QueryConfig(prompt=PromptConfig(user_template="{sources} {question}")),
            QueryConfig(generator=GeneratorConfig(name="llama3.2:3b")),
            QueryConfig(generator=GeneratorConfig(temperature=0.7)),
        ],
    )
    def test_when_any_hyperparameter_changes_then_changes(
        self, package: Path, config: QueryConfig
    ) -> None:
        assert compute(config) != compute()

    def test_when_listed_module_missing_then_raises(self, package: Path) -> None:
        (package / "retriever.py").unlink()

        with pytest.raises(FileNotFoundError, match="retriever.py"):
            compute()
