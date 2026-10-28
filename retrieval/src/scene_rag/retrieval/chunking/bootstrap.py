"""Composition root of the chunking step.

Binds settings from the environment and hyperparameters from `ChunkingConfig`, connects the
scene-api and the Chunk Store, wires the chunker into `ChunkingService`, and that into a
`Scheduler` under the dispatcher the settings call for. Nothing else reads the environment,
and nothing else names a concurrency policy or a chunking strategy.
"""

import contextlib
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Final

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.retrieval.chunking import settings as settings_module
from scene_rag.retrieval.chunking.chunk_store import Document, MongoChunkStore
from scene_rag.retrieval.chunking.config import ChunkingConfig
from scene_rag.retrieval.chunking.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.retrieval.chunking.http_transport import JsonTransport, UrllibTransport
from scene_rag.retrieval.chunking.models import truncate_to_millisecond
from scene_rag.retrieval.chunking.per_scene import PerSceneChunker
from scene_rag.retrieval.chunking.protocols import (
    Chunker,
    ChunkStoreWriter,
    Dispatcher,
    SceneApiError,
    SceneRecordReader,
)
from scene_rag.retrieval.chunking.scene_api_client import SceneApiClient
from scene_rag.retrieval.chunking.scheduler import Scheduler
from scene_rag.retrieval.chunking.service import ChunkingService
from scene_rag.retrieval.chunking.settings import ChunkingSettings

logger = logging.getLogger(__name__)

# Operational guardrails, not tunables: the connectivity check's patience, and the longest
# pause between retries once passes keep failing.
_SERVER_SELECTION_TIMEOUT_MS: Final = 2000
_MAX_BACKOFF_SECONDS: Final = 300.0
_MAX_BACKOFF_DOUBLINGS: Final = 32  # 2**32 s is past any cap; higher would overflow

type ClientFactory = Callable[..., MongoClient[Document]]


class SystemClock:
    """`Clock` over the system time, truncated to the millisecond the stores keep."""

    def now(self) -> datetime:
        """Return the current UTC time at millisecond precision."""
        return truncate_to_millisecond(datetime.now(UTC))


def _require(environ: Mapping[str, str], name: str, hint: str) -> str:
    value = environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. {hint}")
    return value


def _number(environ: Mapping[str, str], name: str) -> float | None:
    """Read an optional number; ``None`` when unset.

    Raises:
        ValueError: Naming the variable, when the value is not a number.
    """
    value = environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number, not {value!r}") from None


def _count(environ: Mapping[str, str], name: str, *, minimum: int) -> int | None:
    """Read an optional whole number of at least ``minimum``; ``None`` when unset.

    Raises:
        ValueError: Naming the variable, when the value is not such a number.
    """
    value = environ.get(name)
    if not value:
        return None
    message = f"{name} must be a whole number of at least {minimum}, not {value!r}"
    try:
        count = int(value)
    except ValueError:
        raise ValueError(message) from None
    if count < minimum:
        raise ValueError(message)
    return count


def load_settings(environ: Mapping[str, str] | None = None) -> ChunkingSettings:
    """Bind `ChunkingSettings` from the environment.

    The poll interval is optional: a single pass never sleeps. `run_worker` demands it before
    looping. The scene-api timeout and the worker count have defaults.

    Args:
        environ: Variables to read; the process environment by default.

    Returns:
        Validated settings.

    Raises:
        RuntimeError: When a variable is missing or the settings do not validate.
    """
    env = os.environ if environ is None else environ
    try:
        timeout = _number(env, settings_module.SCENE_API_TIMEOUT_SECONDS_ENV)
        poll_interval = _number(env, settings_module.POLL_INTERVAL_SECONDS_ENV)
        workers = _count(env, settings_module.WORKERS_ENV, minimum=1)
        return ChunkingSettings(
            scene_api_url=_require(
                env,
                settings_module.SCENE_API_URL_ENV,
                "Point this at the extraction service's scene-api, e.g. http://localhost:8080.",
            ),
            chunk_store_uri=_require(
                env,
                settings_module.CHUNK_STORE_URI_ENV,
                "Start MongoDB and point this at it, e.g. mongodb://localhost:27017.",
            ),
            chunk_store_database=_require(
                env, settings_module.CHUNK_STORE_DATABASE_ENV, "Name the chunk store database."
            ),
            scene_api_timeout_seconds=(
                settings_module.DEFAULT_SCENE_API_TIMEOUT_SECONDS if timeout is None else timeout
            ),
            poll_interval_seconds=poll_interval,
            workers=1 if workers is None else workers,
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid settings: {exc}") from exc


def connect_mongo(
    uri: str, *, redacted: str, client_factory: ClientFactory = MongoClient
) -> MongoClient[Document]:
    """Connect to a MongoDB and verify it answers.

    Args:
        uri: The connection string.
        redacted: The connection string as it may appear in logs and errors.
        client_factory: Builds the client from the connection string and keyword options;
            `MongoClient` itself outside tests.

    Returns:
        A connected client; the caller owns it and closes it, e.g. with ``with``.

    Raises:
        RuntimeError: When the connection string is malformed or the server cannot be
            reached.
    """
    try:
        client = client_factory(
            uri, tz_aware=True, serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS
        )
        client.admin.command("ping")
    except (PyMongoError, ValueError) as exc:  # pymongo raises ValueError for a bad port
        raise RuntimeError(
            f"Could not reach MongoDB at {redacted!r}: {exc}. Check that it is running and the "
            "connection string is correct."
        ) from exc
    logger.info("connected to MongoDB at %s", redacted)
    return client


def connect_scene_api(
    settings: ChunkingSettings, transport: JsonTransport | None = None
) -> SceneApiClient:
    """Build the scene-api client and check that the API answers, before any pass runs.

    Args:
        settings: Where the API is and how long to wait for it.
        transport: The HTTP layer; the standard library outside tests.

    Raises:
        RuntimeError: When the API does not answer; every pass would fail otherwise.
    """
    client = SceneApiClient(
        settings.scene_api_url,
        UrllibTransport() if transport is None else transport,
        settings.scene_api_timeout_seconds,
    )
    try:
        client.health()
    except SceneApiError as exc:
        raise RuntimeError(
            f"The scene-api is not reachable at {settings.redacted_scene_api_url}: {exc}. "
            "Start it (`pdm run scene-api` in the extraction service, or `docker compose up`) "
            f"or point {settings_module.SCENE_API_URL_ENV} at it."
        ) from exc
    logger.info("connected to scene-api at %s", settings.redacted_scene_api_url)
    return client


def create_chunker(config: ChunkingConfig | None = None) -> Chunker:
    """Return the chunking strategy in force: the per-scene chunker.

    The registration point of the `Chunker` extension point: another strategy is composed
    here, never selected by a flag elsewhere.

    Args:
        config: Hyperparameters; the defaults when ``None``.
    """
    return PerSceneChunker(ChunkingConfig() if config is None else config)


def create_service(
    reader: SceneRecordReader,
    writer: ChunkStoreWriter,
    chunker: Chunker,
    config: ChunkingConfig | None = None,
) -> ChunkingService:
    """Assemble the per-scene-set unit of work.

    Args:
        reader: Read view of the scene-api.
        writer: Command side of the Chunk Store.
        chunker: The chunking strategy.
        config: Hyperparameters; the defaults when ``None``.
    """
    return ChunkingService(
        reader=reader,
        writer=writer,
        chunker=chunker,
        clock=SystemClock(),
        config=ChunkingConfig() if config is None else config,
    )


def create_scheduler(
    settings: ChunkingSettings,
    chunk_client: MongoClient[Document],
    reader: SceneRecordReader,
    *,
    chunker: Chunker,
    dispatcher: Dispatcher | None = None,
    config: ChunkingConfig | None = None,
) -> Scheduler:
    """Assemble a pass: the Chunk Store and the scene-api view around `create_service`.

    Args:
        settings: Where the step runs.
        chunk_client: Connection to the Chunk Store.
        reader: Read view of the scene-api, from `connect_scene_api` or a fake.
        chunker: The chunking strategy, from `create_chunker` or a fake.
        dispatcher: Policy the scene sets of a pass run under; serial when ``None``.
        config: Hyperparameters; the defaults when ``None``.
    """
    config = ChunkingConfig() if config is None else config
    store = MongoChunkStore(chunk_client, settings.chunk_store_database)
    return Scheduler(
        reader=reader,
        chunk_reader=store,
        chunker=create_service(reader, store, chunker, config),
        dispatcher=SerialDispatcher() if dispatcher is None else dispatcher,
        config=config,
    )


def make_dispatcher(workers: int, stack: contextlib.ExitStack, *, name: str) -> Dispatcher:
    """Return the concurrency policy for ``workers``: serial for one, a thread pool above.

    A pool is registered on ``stack``, so it drains before the stack's stores close.

    Args:
        workers: Calls in flight at once.
        stack: Owner of the pool's lifetime.
        name: Prefix of the pool's thread names, for logs and tracebacks.
    """
    if workers == 1:
        return SerialDispatcher()
    return stack.enter_context(ThreadDispatcher(workers, thread_name_prefix=name))


def _backoff_seconds(interval: float, failures: int) -> float:
    """Pause after ``failures`` consecutive failed passes: doubling from the interval, capped.

    The exponent is capped too: ``2.0 ** 1024`` overflows, and a worker that has retried for
    days must keep retrying rather than die.
    """
    cap = max(interval, _MAX_BACKOFF_SECONDS)
    return min(interval * 2.0 ** min(failures, _MAX_BACKOFF_DOUBLINGS), cap)


def run_worker(*, once: bool, sleep: Callable[[float], None] = time.sleep) -> None:
    """Entry point of the ``chunking`` step in the worker role.

    The Chunk Store is connected first, then the scene-api is probed; a wrong connection
    string is reported before the first pass. A pass that raises (a store outage, say) ends
    the process in ``once`` mode. In the loop it is logged and the next pass runs after a
    pause that doubles with every consecutive failure, from the poll interval up to a few
    minutes.

    Args:
        once: Run a single pass instead of looping until interrupted.
        sleep: Pause function between passes; injectable for tests.

    Raises:
        RuntimeError: When settings are missing, the loop has no poll interval, the
            scene-api does not answer, or the store is unreachable.
    """
    settings = load_settings()
    interval = settings.poll_interval_seconds
    if not once and interval is None:
        raise RuntimeError(
            f"{settings_module.POLL_INTERVAL_SECONDS_ENV} is not set. "
            "Give the pause between passes in seconds, or run with --once."
        )
    with contextlib.ExitStack() as stack:
        chunk_client = stack.enter_context(
            connect_mongo(settings.chunk_store_uri, redacted=settings.redacted_chunk_store_uri)
        )
        reader = connect_scene_api(settings)
        chunker = create_chunker()
        scheduler = create_scheduler(
            settings,
            chunk_client,
            reader,
            chunker=chunker,
            dispatcher=make_dispatcher(settings.workers, stack, name="chunking"),
        )
        logger.info(
            "chunking fingerprint %s (%s); %d scene set(s) at once",
            scheduler.fingerprint,
            chunker.name,
            settings.workers,
        )
        if once:
            scheduler.run_once()
            return
        assert interval is not None  # checked above, before connecting
        failures = 0
        while True:
            try:
                scheduler.run_once()
            except Exception:
                pause = _backoff_seconds(interval, failures)
                failures += 1
                logger.exception("chunking pass failed; retrying in %.1f s", pause)
            else:
                pause = interval
                failures = 0
            sleep(pause)
