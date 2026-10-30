import argparse

import pytest

from scene_rag.evaluation import __main__ as cli
from scene_rag.evaluation import bootstrap
from scene_rag.evaluation.models import Verdict


class TestGivenCommandLine:
    """The `run` and `compare` subcommands of the service CLI."""

    def test_when_commands_listed_then_every_command_registered(self) -> None:
        assert sorted(cli.COMMANDS) == ["compare", "run"]

    def test_when_command_missing_then_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_when_dataset_missing_then_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["run"])

    def test_when_run_dispatched_then_bootstrap_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []
        monkeypatch.setattr(bootstrap, "run", calls.append)

        assert cli.main(["run", "--dataset", "data/example/1.0.0.jsonl"]) == 0
        assert [str(path) for path in calls] == ["data/example/1.0.0.jsonl"]

    @pytest.mark.parametrize(("passed", "code"), [(True, 0), (False, cli.GATE_FAILED)])
    def test_when_compare_dispatched_then_exit_code_follows_verdict(
        self, monkeypatch: pytest.MonkeyPatch, passed: bool, code: int
    ) -> None:
        class Decided:
            def __init__(self) -> None:
                self.passed = passed

        calls: list[tuple[str, str | None]] = []

        def compare(candidate: str, baseline: str | None) -> Verdict:
            calls.append((candidate, baseline))
            return Decided()  # type: ignore[return-value]

        monkeypatch.setattr(bootstrap, "compare", compare)

        assert cli.main(["compare", "--candidate", "run-000000000002"]) == code
        assert calls == [("run-000000000002", None)]

    def test_when_command_raises_runtime_error_then_exit_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(args: argparse.Namespace) -> int:
            raise RuntimeError("query-api does not answer")

        monkeypatch.setitem(cli.COMMANDS, "run", broken)

        assert cli.main(["run", "--dataset", "x"]) == 1

    def test_when_interrupted_then_exit_interrupted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def interrupted(args: argparse.Namespace) -> int:
            raise KeyboardInterrupt

        monkeypatch.setitem(cli.COMMANDS, "compare", interrupted)

        assert cli.main(["compare", "--candidate", "run-000000000002"]) == cli.INTERRUPTED
