"""Command line of the `retrieval` service.

Roles are subcommands registered in `ROLES`; steps of the worker role are registered in
`STEPS`. Neither is selected by ``if``/``elif``. The ``pull-models`` role puts the embedding
step's model and the query-api role's generator on the Ollama server and exits.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
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


def _query_api_bootstrap() -> ModuleType:
    # Imported on dispatch: the role's extra (FastAPI, uvicorn, the MCP SDK) is needed only
    # to run it.
    try:
        from scene_rag.retrieval.query_api import bootstrap
    except ImportError as exc:
        raise RuntimeError(
            f"The query-api role cannot be imported: {exc}. "
            "Install its extra with `pdm install -G query-api`."
        ) from exc
    return bootstrap


def _query_api(args: argparse.Namespace) -> None:
    entitlements = frozenset(
        name.strip() for name in (args.entitlements or "").split(",") if name.strip()
    )
    _query_api_bootstrap().run_server(transport=args.transport, entitlements=entitlements)


def _pull_models(args: argparse.Namespace) -> None:
    from scene_rag.retrieval.embedding import bootstrap

    bootstrap.pull_models()
    _query_api_bootstrap().pull_models()


ROLES: Final[Mapping[str, RoleRunner]] = {
    "worker": _worker,
    "query-api": _query_api,
    "pull-models": _pull_models,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Subcommands: ``worker --step <step> [--once]``, ``query-api [--transport http|stdio]
    [--entitlements a,b]`` and ``pull-models``.
    """
    parser = argparse.ArgumentParser(prog="scene-rag-retrieval")
    roles = parser.add_subparsers(dest="role", required=True)
    worker = roles.add_parser("worker", help="run one step's worker loop")
    worker.add_argument("--step", choices=sorted(STEPS), required=True)
    worker.add_argument("--once", action="store_true", help="run a single pass and exit")
    query_api = roles.add_parser("query-api", help="serve search and answers over HTTP and MCP")
    query_api.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="http: REST routes and the MCP transport on one port; stdio: MCP for a local client",
    )
    query_api.add_argument(
        "--entitlements",
        default=None,
        help="over stdio, the caller's comma-separated entitlements; over HTTP they come from "
        "the X-Scene-Rag-Entitlements header",
    )
    roles.add_parser(
        "pull-models",
        help="pull the embedding model and the generator into the Ollama server, then exit",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.role == "query-api" and args.transport == "http" and args.entitlements is not None:
        parser.error("--entitlements applies to --transport stdio only")
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
