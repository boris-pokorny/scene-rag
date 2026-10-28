import importlib
import subprocess
import sys

import pytest

from scene_rag.extraction import __main__ as cli


class TestGivenCommandLine:
    """The `worker` subcommand of the service CLI."""

    def test_when_step_unknown_then_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.build_parser().parse_args(["worker", "--step", "teleportation"])
        assert exit_info.value.code == 2
        assert "ingestion" in capsys.readouterr().err

    def test_when_steps_listed_then_every_step_registered(self) -> None:
        assert sorted(cli.STEPS) == ["enrichment", "ingestion", "segmentation"]

    def test_when_roles_listed_then_every_role_registered(self) -> None:
        assert sorted(cli.ROLES) == ["pull-models", "scene-api", "worker"]

    def test_when_scene_api_role_given_then_dispatched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setitem(cli.ROLES, "scene-api", lambda args: calls.append(args.role))

        assert cli.main(["scene-api"]) == 0
        assert calls == ["scene-api"]

    @pytest.mark.parametrize("role", ["scene-api", "pull-models"])
    def test_when_role_given_worker_options_then_rejected(self, role: str) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([role, "--once"])

    def test_when_pull_models_role_given_then_dispatched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setitem(cli.ROLES, "pull-models", lambda args: calls.append(args.role))

        assert cli.main(["pull-models"]) == 0
        assert calls == ["pull-models"]

    def test_when_role_missing_then_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_when_step_registered_then_dispatched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[bool] = []
        monkeypatch.setitem(cli.STEPS, "ingestion", lambda *, once: calls.append(once))

        assert cli.main(["worker", "--step", "ingestion", "--once"]) == 0
        assert calls == [True]

    def test_when_interrupted_then_exit_code_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def interrupt(*, once: bool) -> None:
            raise KeyboardInterrupt

        monkeypatch.setitem(cli.STEPS, "ingestion", interrupt)

        assert cli.main(["worker", "--step", "ingestion"]) == 0

    def test_when_configuration_error_then_exit_code_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*, once: bool) -> None:
            raise RuntimeError("SCENE_RAG_EXTRACTION_REGISTRY_URI is not set.")

        monkeypatch.setitem(cli.STEPS, "ingestion", fail)

        assert cli.main(["worker", "--step", "ingestion"]) == 1


class TestGivenStepExtras:
    """Each step's and role's dependencies load only when it is dispatched."""

    @pytest.mark.parametrize("package", ["scenedetect", "fastapi"])
    def test_when_cli_imported_without_extra_then_import_succeeds(self, package: str) -> None:
        code = (
            "import sys\n"
            f"sys.modules[{package!r}] = None\n"  # makes `import <package>` fail
            "import scene_rag.extraction.__main__\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("step", ["segmentation", "enrichment"])
    def test_when_step_extra_missing_then_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, step: str
    ) -> None:
        package = importlib.import_module(f"scene_rag.extraction.{step}")

        # `None` in sys.modules makes the import fail; the package attribute set by an earlier
        # import would otherwise satisfy `from ... import bootstrap` without importing.
        monkeypatch.setitem(sys.modules, f"scene_rag.extraction.{step}.bootstrap", None)
        monkeypatch.delattr(package, "bootstrap", raising=False)

        with pytest.raises(RuntimeError, match=f"pdm install -G {step}"):
            cli.STEPS[step](once=True)

    @pytest.mark.parametrize(
        ("role", "package", "extra"),
        [("scene-api", "scene_api", "scene-api"), ("pull-models", "enrichment", "enrichment")],
    )
    def test_when_role_extra_missing_then_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, role: str, package: str, extra: str
    ) -> None:
        module = importlib.import_module(f"scene_rag.extraction.{package}")
        monkeypatch.setitem(sys.modules, f"scene_rag.extraction.{package}.bootstrap", None)
        monkeypatch.delattr(module, "bootstrap", raising=False)

        with pytest.raises(RuntimeError, match=f"pdm install -G {extra}"):
            cli.ROLES[role](cli.build_parser().parse_args([role]))
