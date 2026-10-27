"""Command line of the `extraction` service.

Roles are subcommands; steps of the worker role are registered in `STEPS`, never selected
by ``if``/``elif``.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from scene_rag.extraction.ingestion import bootstrap as ingestion

type StepRunner = Callable[..., None]

STEPS: Final[Mapping[str, StepRunner]] = {"ingestion": ingestion.run_worker}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser: ``worker --step <step> [--once]``."""
    parser = argparse.ArgumentParser(prog="scene-rag-extraction")
    roles = parser.add_subparsers(dest="role", required=True)
    worker = roles.add_parser("worker", help="run one step's worker loop")
    worker.add_argument("--step", choices=sorted(STEPS), required=True)
    worker.add_argument("--once", action="store_true", help="run a single pass and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the selected role.

    Returns:
        Process exit code: 0 on success or interruption, 1 on a configuration error.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        STEPS[args.step](once=args.once)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("interrupted")
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
