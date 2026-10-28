"""Command line of the `extraction` service.

Roles are subcommands registered in `ROLES`; steps of the worker role are registered in
`STEPS`. Neither is selected by ``if``/``elif``.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Final

type StepRunner = Callable[..., None]
type RoleRunner = Callable[[argparse.Namespace], None]


def _ingestion(*, once: bool) -> None:
    from scene_rag.extraction.ingestion import bootstrap

    bootstrap.run_worker(once=once)


def _segmentation(*, once: bool) -> None:
    # Imported on dispatch: the step's extra (PySceneDetect, OpenCV) is needed only to run it.
    try:
        from scene_rag.extraction.segmentation import bootstrap
    except ImportError as exc:
        raise RuntimeError(
            f"The segmentation step cannot be imported: {exc}. "
            "Install its extra with `pdm install -G segmentation`."
        ) from exc
    bootstrap.run_worker(once=once)


def _enrichment(*, once: bool) -> None:
    # Imported on dispatch: the step's extra (faster-whisper) is needed only to run it.
    try:
        from scene_rag.extraction.enrichment import bootstrap
    except ImportError as exc:
        raise RuntimeError(
            f"The enrichment step cannot be imported: {exc}. "
            "Install its extra with `pdm install -G enrichment`."
        ) from exc
    bootstrap.run_worker(once=once)


# Each step's bootstrap is imported inside its runner, so a role process loads only the
# step it runs and the dependencies that step needs.
STEPS: Final[Mapping[str, StepRunner]] = {
    "ingestion": _ingestion,
    "segmentation": _segmentation,
    "enrichment": _enrichment,
}


def _worker(args: argparse.Namespace) -> None:
    STEPS[args.step](once=args.once)


def _scene_api(args: argparse.Namespace) -> None:
    # Imported on dispatch: the role's extra (FastAPI, uvicorn) is needed only to run it.
    try:
        from scene_rag.extraction.scene_api import bootstrap
    except ImportError as exc:
        raise RuntimeError(
            f"The scene-api role cannot be imported: {exc}. "
            "Install its extra with `pdm install -G scene-api`."
        ) from exc
    bootstrap.run_server()


ROLES: Final[Mapping[str, RoleRunner]] = {
    "worker": _worker,
    "scene-api": _scene_api,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser: ``worker --step <step> [--once]`` or ``scene-api``."""
    parser = argparse.ArgumentParser(prog="scene-rag-extraction")
    roles = parser.add_subparsers(dest="role", required=True)
    worker = roles.add_parser("worker", help="run one step's worker loop")
    worker.add_argument("--step", choices=sorted(STEPS), required=True)
    worker.add_argument("--once", action="store_true", help="run a single pass and exit")
    roles.add_parser(
        "scene-api", help="serve the scene records over HTTP; settings from the environment"
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
