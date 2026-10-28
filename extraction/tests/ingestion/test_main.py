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

    def test_when_steps_listed_then_both_registered(self) -> None:
        assert sorted(cli.STEPS) == ["ingestion", "segmentation"]

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
    """Each step's dependencies load only when that step is dispatched."""

    def test_when_cli_imported_without_scenedetect_then_import_succeeds(self) -> None:
        code = (
            "import sys\n"
            "sys.modules['scenedetect'] = None\n"  # makes `import scenedetect` fail
            "import scene_rag.extraction.__main__\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr

    def test_when_segmentation_extra_missing_then_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scene_rag.extraction.segmentation as package

        # `None` in sys.modules makes the import fail; the package attribute set by an earlier
        # import would otherwise satisfy `from ... import bootstrap` without importing.
        monkeypatch.setitem(sys.modules, "scene_rag.extraction.segmentation.bootstrap", None)
        monkeypatch.delattr(package, "bootstrap", raising=False)

        with pytest.raises(RuntimeError, match="pdm install -G segmentation"):
            cli.STEPS["segmentation"](once=True)
