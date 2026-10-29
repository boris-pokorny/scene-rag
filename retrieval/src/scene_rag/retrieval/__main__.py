"""Command line of the `retrieval` service.

Roles are subcommands registered in `ROLES`; steps of the worker role are registered in
`STEPS`. Neither is selected by ``if``/``elif``. The ``pull-models`` role puts the embedding
step's model on the Ollama server and exits.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Final

type StepRunner = Callable[..., None]
type RoleRunner = Callable[[argparse.Namespace], None]


def _chunking(*, once: bool) -> None:
    from scene_rag.retrieval.chunking import bootstrap

    bootstrap.run_worker(once=once)


def _embedding(*, once: bool) -> None:
    from scene_rag.retrieval.embedding import bootstrap

    bootstrap.run_worker(once=once)


# Each step's bootstrap is imported inside its runner, so a role process loads only the
# step it runs and the dependencies that step needs.
STEPS: Final[Mapping[str, StepRunner]] = {
    "chunking": _chunking,
    "embedding": _embedding,
}


def _worker(args: argparse.Namespace) -> None:
    STEPS[args.step](once=args.once)


def _pull_models(args: argparse.Namespace) -> None:
    from scene_rag.retrieval.embedding import bootstrap

    bootstrap.pull_models()


ROLES: Final[Mapping[str, RoleRunner]] = {
    "worker": _worker,
    "pull-models": _pull_models,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Subcommands: ``worker --step <step> [--once]`` and ``pull-models``.
    """
    parser = argparse.ArgumentParser(prog="scene-rag-retrieval")
    roles = parser.add_subparsers(dest="role", required=True)
    worker = roles.add_parser("worker", help="run one step's worker loop")
    worker.add_argument("--step", choices=sorted(STEPS), required=True)
    worker.add_argument("--once", action="store_true", help="run a single pass and exit")
    roles.add_parser(
        "pull-models",
        help="pull the embedding step's model into the Ollama server, then exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the selected role.

    Returns:
        Process exit code: 0 on success or interruption, 1 on a configuration error.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        ROLES[args.role](args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("interrupted")
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
