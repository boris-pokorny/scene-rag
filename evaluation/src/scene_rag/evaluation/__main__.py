"""Command line of the `evaluation` service.

Commands are subcommands registered in `COMMANDS`, never selected by ``if``/``elif``. Both
are one-shot processes of one image, so they share one composition root.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from scene_rag.evaluation import bootstrap

type CommandRunner = Callable[[argparse.Namespace], int]
"""Runs a command and returns the process exit code."""

GATE_FAILED: Final = 3
"""Exit code of a compare whose verdict is ``failed``; 1 is a configuration error, 2 a usage
error (argparse)."""

INTERRUPTED: Final = 130
"""Exit code of an interrupted command (128 + SIGINT); never 0, so it cannot read as a pass."""


def _run(args: argparse.Namespace) -> int:
    bootstrap.run(Path(args.dataset))
    return 0


def _compare(args: argparse.Namespace) -> int:
    verdict = bootstrap.compare(args.candidate, args.baseline)
    return 0 if verdict.passed else GATE_FAILED


COMMANDS: Final[Mapping[str, CommandRunner]] = {
    "run": _run,
    "compare": _compare,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Subcommands: ``run --dataset <path>`` and ``compare --candidate <run id>
    [--baseline <run id>]``.
    """
    parser = argparse.ArgumentParser(prog="scene-rag-evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate query-api against a dataset, store the run")
    run.add_argument(
        "--dataset", required=True, help="the dataset file, <name>/<major.minor.patch>.jsonl"
    )
    compare = commands.add_parser("compare", help="diff two runs and decide the promotion gate")
    compare.add_argument("--candidate", required=True, help="id of the run to decide on")
    compare.add_argument(
        "--baseline",
        default=None,
        help="id of the active release's run; without it only the absolute rules apply",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the selected command.

    Returns:
        Process exit code: 0 on success, 1 on a configuration error, 2 on a usage error,
        `GATE_FAILED` when the gate fails, `INTERRUPTED` on interruption.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("interrupted")
        return INTERRUPTED
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
