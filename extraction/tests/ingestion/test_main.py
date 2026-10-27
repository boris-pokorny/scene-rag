import pytest

from scene_rag.extraction import __main__ as cli


class TestGivenCommandLine:
    """The `worker` subcommand of the service CLI."""

    def test_when_step_unknown_then_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.build_parser().parse_args(["worker", "--step", "teleportation"])
        assert exit_info.value.code == 2
        assert "ingestion" in capsys.readouterr().err

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
