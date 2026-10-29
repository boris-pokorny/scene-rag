import subprocess
import sys

import pytest

from scene_rag.retrieval import __main__ as cli


class TestGivenCommandLine:
    """The `worker`, `query-api` and `pull-models` subcommands of the service CLI."""

    def test_when_step_unknown_then_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.build_parser().parse_args(["worker", "--step", "teleportation"])
        assert exit_info.value.code == 2
        assert "chunking" in capsys.readouterr().err

    def test_when_steps_listed_then_every_step_registered(self) -> None:
        assert sorted(cli.STEPS) == ["chunking", "embedding"]

    def test_when_roles_listed_then_every_role_registered(self) -> None:
        assert sorted(cli.ROLES) == ["pull-models", "query-api", "worker"]

    def test_when_role_missing_then_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    @pytest.mark.parametrize("step", ["chunking", "embedding"])
    def test_when_step_registered_then_dispatched(
        self, monkeypatch: pytest.MonkeyPatch, step: str
    ) -> None:
        calls: list[bool] = []
        monkeypatch.setitem(cli.STEPS, step, lambda *, once: calls.append(once))

        assert cli.main(["worker", "--step", step, "--once"]) == 0
        assert calls == [True]

    def test_when_chunking_dispatched_then_bootstrap_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scene_rag.retrieval.chunking import bootstrap

        calls: list[bool] = []
        monkeypatch.setattr(bootstrap, "run_worker", lambda *, once: calls.append(once))

        assert cli.main(["worker", "--step", "chunking"]) == 0
        assert calls == [False]

    def test_when_embedding_dispatched_then_bootstrap_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scene_rag.retrieval.embedding import bootstrap

        calls: list[bool] = []
        monkeypatch.setattr(bootstrap, "run_worker", lambda *, once: calls.append(once))

        assert cli.main(["worker", "--step", "embedding"]) == 0
        assert calls == [False]

    def test_when_pull_models_dispatched_then_both_bootstraps_pull(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scene_rag.retrieval.embedding import bootstrap as embedding
        from scene_rag.retrieval.query_api import bootstrap as query_api

        calls: list[str] = []
        monkeypatch.setattr(embedding, "pull_models", lambda: calls.append("embedding"))
        monkeypatch.setattr(query_api, "pull_models", lambda: calls.append("query-api"))

        assert cli.main(["pull-models"]) == 0
        assert calls == ["embedding", "query-api"]

    @pytest.mark.parametrize(
        "argv, transport, entitlements",
        [
            (["query-api"], "http", frozenset()),
            (["query-api", "--transport", "http"], "http", frozenset()),
            (["query-api", "--transport", "stdio"], "stdio", frozenset()),
            (
                ["query-api", "--transport", "stdio", "--entitlements", "public, team ,"],
                "stdio",
                frozenset({"public", "team"}),
            ),
        ],
    )
    def test_when_query_api_dispatched_then_bootstrap_serves(
        self,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        transport: str,
        entitlements: frozenset[str],
    ) -> None:
        from scene_rag.retrieval.query_api import bootstrap

        calls: list[tuple[str, frozenset[str]]] = []
        monkeypatch.setattr(
            bootstrap,
            "run_server",
            lambda *, transport, entitlements: calls.append((transport, entitlements)),
        )

        assert cli.main(argv) == 0
        assert calls == [(transport, entitlements)]

    def test_when_entitlements_given_over_http_then_rejected(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["query-api", "--entitlements", "public"])
        assert exit_info.value.code == 2
        assert "stdio" in capsys.readouterr().err

    def test_when_query_api_extra_missing_then_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def failing_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "scene_rag.retrieval.query_api":
                raise ImportError("No module named 'fastapi'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", failing_import)

        assert cli.main(["query-api"]) == 1

    def test_when_interrupted_then_exit_code_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def interrupt(*, once: bool) -> None:
            raise KeyboardInterrupt

        monkeypatch.setitem(cli.STEPS, "chunking", interrupt)

        assert cli.main(["worker", "--step", "chunking"]) == 0

    def test_when_configuration_error_then_exit_code_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*, once: bool) -> None:
            raise RuntimeError("SCENE_RAG_RETRIEVAL_SCENE_API_URL is not set.")

        monkeypatch.setitem(cli.STEPS, "chunking", fail)

        assert cli.main(["worker", "--step", "chunking"]) == 1

    def test_when_cli_imported_then_no_step_loaded(self) -> None:
        code = "import sys\nimport scene_rag.retrieval.__main__\n" + (
            "assert 'scene_rag.retrieval.chunking.bootstrap' not in sys.modules\n"
            "assert 'scene_rag.retrieval.embedding.bootstrap' not in sys.modules\n"
            "assert 'scene_rag.retrieval.query_api' not in sys.modules\n"
            "assert 'fastapi' not in sys.modules and 'mcp' not in sys.modules"
        )

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
