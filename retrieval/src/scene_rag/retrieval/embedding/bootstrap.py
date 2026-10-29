"""Composition root of the embedding step.

Binds settings from the environment and hyperparameters from `EmbeddingConfig`, connects the
Chunk Store, the index ledger, Ollama and the Search Index, wires the model into
`EmbeddingService`, and that into a `Scheduler` under the dispatcher the settings call for.
Nothing else reads the environment, and nothing else names a backend or a model.
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

from scene_rag.retrieval.embedding import ollama_pull
from scene_rag.retrieval.embedding import settings as settings_module
from scene_rag.retrieval.embedding.chunk_reader import Document, MongoChunkReader
from scene_rag.retrieval.embedding.config import EmbeddingConfig
from scene_rag.retrieval.embedding.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.retrieval.embedding.http_transport import JsonTransport, UrllibTransport
from scene_rag.retrieval.embedding.index_setup import prepare_index
from scene_rag.retrieval.embedding.index_store import MongoIndexStore
from scene_rag.retrieval.embedding.models import truncate_to_millisecond
from scene_rag.retrieval.embedding.ollama_embedding import OllamaClient, OllamaEmbeddingModel
from scene_rag.retrieval.embedding.protocols import (
    Dispatcher,
    EmbeddingModel,
    ModelError,
    SearchIndex,
    SearchIndexError,
)
from scene_rag.retrieval.embedding.qdrant_index import QdrantSearchIndex
from scene_rag.retrieval.embedding.scheduler import Scheduler
from scene_rag.retrieval.embedding.service import EmbeddingService
from scene_rag.retrieval.embedding.settings import EmbeddingSettings, OllamaSettings

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


def load_ollama_settings(environ: Mapping[str, str] | None = None) -> OllamaSettings:
    """Bind `OllamaSettings` from the environment; both variables have defaults.

    Raises:
        RuntimeError: When a value does not validate.
    """
    env = os.environ if environ is None else environ
    try:
        timeout = _number(env, settings_module.OLLAMA_TIMEOUT_SECONDS_ENV)
        return OllamaSettings(
            url=env.get(settings_module.OLLAMA_URL_ENV) or settings_module.DEFAULT_OLLAMA_URL,
            timeout_seconds=(
                settings_module.DEFAULT_OLLAMA_TIMEOUT_SECONDS if timeout is None else timeout
            ),
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid settings: {exc}") from exc


def load_settings(environ: Mapping[str, str] | None = None) -> EmbeddingSettings:
    """Bind `EmbeddingSettings` from the environment.

    The poll interval is optional: a single pass never sleeps. `run_worker` demands it before
    looping. Timeouts, batch size and worker count have defaults.

    Args:
        environ: Variables to read; the process environment by default.

    Returns:
        Validated settings.

    Raises:
        RuntimeError: When a variable is missing or the settings do not validate.
    """
    env = os.environ if environ is None else environ
    ollama = load_ollama_settings(env)
    try:
        timeout = _number(env, settings_module.SEARCH_INDEX_TIMEOUT_SECONDS_ENV)
        poll_interval = _number(env, settings_module.POLL_INTERVAL_SECONDS_ENV)
        batch_size = _count(env, settings_module.BATCH_SIZE_ENV, minimum=1)
        workers = _count(env, settings_module.WORKERS_ENV, minimum=1)
        return EmbeddingSettings(
            chunk_store_uri=_require(
                env,
                settings_module.CHUNK_STORE_URI_ENV,
                "Start MongoDB and point this at it, e.g. mongodb://localhost:27017.",
            ),
            chunk_store_database=_require(
                env, settings_module.CHUNK_STORE_DATABASE_ENV, "Name the chunk store database."
            ),
            search_index_url=_require(
                env,
                settings_module.SEARCH_INDEX_URL_ENV,
                "Start Qdrant and point this at it, e.g. http://localhost:6333.",
            ),
            ollama=ollama,
            search_index_timeout_seconds=(
                settings_module.DEFAULT_SEARCH_INDEX_TIMEOUT_SECONDS if timeout is None else timeout
            ),
            batch_size=settings_module.DEFAULT_BATCH_SIZE if batch_size is None else batch_size,
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


def connect_ollama(
    settings: OllamaSettings, transport: JsonTransport | None = None
) -> OllamaClient:
    """Build the Ollama client and check that the server answers, before any pass runs.

    Args:
        settings: Where the server is and how long to wait for it.
        transport: The HTTP layer; the standard library outside tests.

    Raises:
        RuntimeError: When the server does not answer; every pass would fail otherwise.
    """
    client = OllamaClient(
        settings.url,
        UrllibTransport() if transport is None else transport,
        settings.timeout_seconds,
    )
    try:
        version = client.version()
    except ModelError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {settings.redacted_url}: {exc}. "
            "Start it (`docker compose up -d ollama`) or point "
            f"{settings_module.OLLAMA_URL_ENV} at it."
        ) from exc
    logger.info("connected to Ollama %s at %s", version, settings.redacted_url)
    return client


def connect_search_index(
    settings: EmbeddingSettings, transport: JsonTransport | None = None
) -> SearchIndex:
    """Build the Search Index backend and check that it answers, before any pass runs.

    The registration point of the backend: another vector database is composed here, never
    selected by a flag elsewhere.

    Args:
        settings: Where the backend is and how long to wait for it.
        transport: The HTTP layer; the standard library outside tests.

    Raises:
        RuntimeError: When the backend does not answer; every pass would fail otherwise.
    """
    index = QdrantSearchIndex(
        settings.search_index_url,
        UrllibTransport() if transport is None else transport,
        settings.search_index_timeout_seconds,
    )
    try:
        version = index.version()
    except SearchIndexError as exc:
        raise RuntimeError(
            f"The Search Index is not reachable at {settings.redacted_search_index_url}: {exc}. "
            "Start Qdrant (`docker compose up -d qdrant`) or point "
            f"{settings_module.SEARCH_INDEX_URL_ENV} at it."
        ) from exc
    logger.info("connected to Qdrant %s at %s", version, settings.redacted_search_index_url)
    return index


def create_embedding_model(
    ollama: OllamaClient, config: EmbeddingConfig | None = None
) -> EmbeddingModel:
    """Return the embedding model in force: the configured model on Ollama, digest resolved.

    The registration point of the `EmbeddingModel` extension point: another gateway is
    composed here, never selected by a flag elsewhere.

    Args:
        ollama: The server.
        config: Hyperparameters; the defaults when ``None``.

    Raises:
        RuntimeError: When the server does not hold the model.
    """
    config = EmbeddingConfig() if config is None else config
    try:
        model = ollama.resolve_model(config.model.name)
    except ModelError as exc:
        raise RuntimeError(f"{exc} Or run the pull-models role.") from exc
    return OllamaEmbeddingModel(ollama, model, config)


def create_scheduler(
    settings: EmbeddingSettings,
    chunk_client: MongoClient[Document],
    search_index: SearchIndex,
    model: EmbeddingModel,
    *,
    dispatcher: Dispatcher | None = None,
    config: EmbeddingConfig | None = None,
) -> Scheduler:
    """Assemble a pass: the stores and the index around `EmbeddingService`.

    Prepares the index of the model's fingerprint first (`index_setup.prepare_index`).

    Args:
        settings: Where the step runs.
        chunk_client: Connection to the Chunk Store and the index ledger.
        search_index: The Search Index backend, from `connect_search_index` or a fake.
        model: The embedding model, from `create_embedding_model` or a fake.
        dispatcher: Policy the chunk sets of a pass run under; serial when ``None``.
        config: Hyperparameters; the defaults when ``None``.

    Raises:
        RuntimeError: When the index under the fingerprint was built differently.
    """
    config = EmbeddingConfig() if config is None else config
    clock = SystemClock()
    reader = MongoChunkReader(chunk_client, settings.chunk_store_database)
    store = MongoIndexStore(chunk_client, settings.chunk_store_database)
    try:
        index = prepare_index(
            model=model,
            index_writer=search_index,
            store_reader=store,
            store_writer=store,
            config=config,
            clock=clock,
        )
    except SearchIndexError as exc:
        raise RuntimeError(f"Could not prepare the index: {exc}") from exc
    service = EmbeddingService(
        reader=reader,
        model=model,
        index_writer=search_index,
        store_writer=store,
        index=index,
        clock=clock,
        config=config,
        batch_size=settings.batch_size,
    )
    return Scheduler(
        chunk_reader=reader,
        store_reader=store,
        embedder=service,
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


def pull_models(
    *,
    environ: Mapping[str, str] | None = None,
    config: EmbeddingConfig | None = None,
    transport: JsonTransport | None = None,
) -> None:
    """Entry point of the ``pull-models`` role: put the embedding model on the Ollama server.

    The model is the one `EmbeddingConfig` names, so a deployment pulls what the embedding
    worker will resolve, without naming it a second time. A model the server already holds is
    not pulled again, so the role needs no network once the model is in place.

    Args:
        environ: Variables to read; the process environment by default.
        config: Hyperparameters; the defaults when ``None``.
        transport: HTTP layer for Ollama; the standard library outside tests.

    Raises:
        RuntimeError: When the settings do not validate, Ollama is unreachable or the pull
            fails.
    """
    settings = load_ollama_settings(environ)
    config = EmbeddingConfig() if config is None else config
    name = config.model.name
    ollama = connect_ollama(settings, transport)
    try:
        model = ollama.resolve_model(name)
    except ModelError:
        logger.info("pulling %s into Ollama at %s", name, settings.redacted_url)
        try:
            ollama_pull.pull_model(
                settings.url, name, UrllibTransport() if transport is None else transport
            )
            model = ollama.resolve_model(name)
        except ModelError as exc:
            raise RuntimeError(str(exc)) from exc
    logger.info("model %s is available as %s", model.name, model.digest)


def run_worker(*, once: bool, sleep: Callable[[float], None] = time.sleep) -> None:
    """Entry point of the ``embedding`` step in the worker role.

    The stores are connected first, then Ollama and the Search Index are probed and the
    model resolved; a wrong connection string or a missing model is reported before the
    first pass. A pass that raises ends the process in ``once`` mode. In the loop it is
    logged and the next pass runs after a pause that doubles with every consecutive failure,
    from the poll interval up to a few minutes.

    Args:
        once: Run a single pass instead of looping until interrupted.
        sleep: Pause function between passes; injectable for tests.

    Raises:
        RuntimeError: When settings are missing, the loop has no poll interval, a backend
            does not answer, the model is missing, or the index was built differently.
    """
    settings = load_settings()
    interval = settings.poll_interval_seconds
    if not once and interval is None:
        raise RuntimeError(
            f"{settings_module.POLL_INTERVAL_SECONDS_ENV} is not set. "
            "Give the pause between passes in seconds, or run with --once."
        )
    # One config for the whole process: the model is fingerprinted under the same values the
    # scheduler selects chunk sets with and the index record is written from.
    config = EmbeddingConfig()
    with contextlib.ExitStack() as stack:
        chunk_client = stack.enter_context(
            connect_mongo(settings.chunk_store_uri, redacted=settings.redacted_chunk_store_uri)
        )
        ollama = connect_ollama(settings.ollama)
        search_index = connect_search_index(settings)
        model = create_embedding_model(ollama, config)
        scheduler = create_scheduler(
            settings,
            chunk_client,
            search_index,
            model,
            dispatcher=make_dispatcher(settings.workers, stack, name="embedding"),
            config=config,
        )
        logger.info(
            "embedding fingerprint %s (%s %s %s); %d chunk set(s) at once, %d text(s) per batch",
            scheduler.fingerprint,
            model.name,
            model.model.name,
            model.model.digest,
            settings.workers,
            settings.batch_size,
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
                logger.exception("embedding pass failed; retrying in %.1f s", pause)
            else:
                pause = interval
                failures = 0
            sleep(pause)
