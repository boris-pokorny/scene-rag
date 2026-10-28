"""Composition root of the enrichment step.

Binds settings from the environment and hyperparameters from `EnrichmentConfig`, resolves the
models behind the enrichers, then wires the read views, the store and the enrichers into
`EnrichmentService`, and that into a `Scheduler` under the dispatcher the settings call for.
Nothing else reads the environment, and nothing else names a concurrency policy.
"""

import contextlib
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.extraction.enrichment import settings as settings_module
from scene_rag.extraction.enrichment import whisper_loader
from scene_rag.extraction.enrichment.asset_registry import Document, MongoAssetVersionReader
from scene_rag.extraction.enrichment.config import EnrichmentConfig
from scene_rag.extraction.enrichment.derivative_reader import FileDerivativeReader
from scene_rag.extraction.enrichment.description import DescriptionEnricher
from scene_rag.extraction.enrichment.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.extraction.enrichment.enrichment_store import MongoEnrichmentStore
from scene_rag.extraction.enrichment.models import truncate_to_millisecond
from scene_rag.extraction.enrichment.ollama_backend import OllamaClient, OllamaVisionModel
from scene_rag.extraction.enrichment.ollama_transport import (
    JsonTransport,
    OllamaError,
    UrllibTransport,
)
from scene_rag.extraction.enrichment.protocols import (
    AssetVersionReader,
    Dispatcher,
    Enricher,
    EnrichmentStoreWriter,
    SceneSetReader,
)
from scene_rag.extraction.enrichment.scene_store import MongoSceneSetReader
from scene_rag.extraction.enrichment.scheduler import Scheduler
from scene_rag.extraction.enrichment.service import EnrichmentService
from scene_rag.extraction.enrichment.settings import EnrichmentSettings
from scene_rag.extraction.enrichment.transcription import TranscriptionEnricher

logger = logging.getLogger(__name__)

# Operational guardrails, not tunables: the connectivity check's patience, and the longest
# pause between retries once passes keep failing.
_SERVER_SELECTION_TIMEOUT_MS: Final = 2000
_MAX_BACKOFF_SECONDS: Final = 300.0

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


def load_settings(environ: Mapping[str, str] | None = None) -> EnrichmentSettings:
    """Bind `EnrichmentSettings` from the environment.

    The poll interval is optional: a single pass never sleeps. `run_worker` demands it before
    looping. The Ollama URL and timeout, the speech model cache and threads, and the worker
    count have defaults.

    Args:
        environ: Variables to read; the process environment by default.

    Returns:
        Validated settings.

    Raises:
        RuntimeError: When a variable is missing or the settings do not validate.
    """
    env = os.environ if environ is None else environ
    mongo_hint = "Start MongoDB and point this at it, e.g. mongodb://localhost:27017."
    try:
        timeout = _number(env, settings_module.OLLAMA_TIMEOUT_SECONDS_ENV)
        poll_interval = _number(env, settings_module.POLL_INTERVAL_SECONDS_ENV)
        threads = _count(env, settings_module.WHISPER_CPU_THREADS_ENV, minimum=0)
        workers = _count(env, settings_module.WORKERS_ENV, minimum=1)
        cache_dir = env.get(settings_module.WHISPER_CACHE_DIR_ENV)
        return EnrichmentSettings(
            registry_uri=_require(env, settings_module.REGISTRY_URI_ENV, mongo_hint),
            registry_database=_require(
                env, settings_module.REGISTRY_DATABASE_ENV, "Name the registry database."
            ),
            scene_store_uri=_require(env, settings_module.SCENE_STORE_URI_ENV, mongo_hint),
            scene_store_database=_require(
                env, settings_module.SCENE_STORE_DATABASE_ENV, "Name the scene store database."
            ),
            enrichment_store_uri=_require(
                env, settings_module.ENRICHMENT_STORE_URI_ENV, mongo_hint
            ),
            enrichment_store_database=_require(
                env,
                settings_module.ENRICHMENT_STORE_DATABASE_ENV,
                "Name the enrichment store database.",
            ),
            ollama_url=env.get(settings_module.OLLAMA_URL_ENV)
            or settings_module.DEFAULT_OLLAMA_URL,
            ollama_timeout_seconds=(
                settings_module.DEFAULT_OLLAMA_TIMEOUT_SECONDS if timeout is None else timeout
            ),
            whisper_cache_dir=None if not cache_dir else Path(cache_dir).resolve(),
            whisper_cpu_threads=0 if threads is None else threads,
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
    settings: EnrichmentSettings, transport: JsonTransport | None = None
) -> OllamaClient:
    """Build the Ollama client and check that the server answers, before any scene is attempted.

    Args:
        settings: Where the server is and how long to wait for it.
        transport: The HTTP layer; the standard library outside tests.

    Raises:
        RuntimeError: When the server does not answer; every scene would fail otherwise.
    """
    client = OllamaClient(
        settings.ollama_url,
        UrllibTransport() if transport is None else transport,
        settings.ollama_timeout_seconds,
    )
    try:
        version = client.version()
    except OllamaError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {settings.ollama_url}: {exc}. Start it (e.g. "
            "`docker run -d -p 11434:11434 docker.io/ollama/ollama`) or point "
            f"{settings_module.OLLAMA_URL_ENV} at it."
        ) from exc
    logger.info("connected to Ollama %s at %s", version, settings.ollama_url)
    return client


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


def create_enrichers(
    settings: EnrichmentSettings,
    config: EnrichmentConfig | None = None,
    *,
    transport: JsonTransport | None = None,
) -> tuple[Enricher, ...]:
    """Resolve the models and assemble the reference enrichers: description, then transcript.

    Args:
        settings: Where the models are.
        config: Hyperparameters; the defaults when ``None``.
        transport: HTTP layer for Ollama; the standard library outside tests.

    Raises:
        RuntimeError: When Ollama is unreachable, a model is missing or the speech model cannot
            be loaded.
    """
    config = EnrichmentConfig() if config is None else config
    ollama = connect_ollama(settings, transport)
    try:
        vision_model = ollama.resolve_model(config.description.model)
    except OllamaError as exc:
        raise RuntimeError(str(exc)) from exc
    transcriber = whisper_loader.load_transcriber(
        config.transcription,
        cache_dir=settings.whisper_cache_dir,
        cpu_threads=settings.whisper_cpu_threads,
    )
    derivatives = FileDerivativeReader()
    return (
        DescriptionEnricher(
            vision=OllamaVisionModel(ollama, vision_model, config.description),
            derivatives=derivatives,
            config=config.description,
        ),
        TranscriptionEnricher(
            transcriber=transcriber, derivatives=derivatives, config=config.transcription
        ),
    )


def create_service(
    assets: AssetVersionReader,
    scene_reader: SceneSetReader,
    writer: EnrichmentStoreWriter,
    enrichers: Sequence[Enricher],
    config: EnrichmentConfig | None = None,
) -> EnrichmentService:
    """Assemble the per-scene-set unit of work.

    Args:
        assets: Read view of the Asset Registry.
        scene_reader: Read view of the Scene Store.
        writer: Command side of the Enrichment Store.
        enrichers: The enrichers to run, in order.
        config: Hyperparameters; the defaults when ``None``.
    """
    return EnrichmentService(
        assets=assets,
        scene_reader=scene_reader,
        writer=writer,
        enrichers=enrichers,
        clock=SystemClock(),
        config=EnrichmentConfig() if config is None else config,
    )


def create_scheduler(
    settings: EnrichmentSettings,
    registry_client: MongoClient[Document],
    scene_client: MongoClient[Document],
    enrichment_client: MongoClient[Document],
    *,
    enrichers: Sequence[Enricher],
    dispatcher: Dispatcher | None = None,
    config: EnrichmentConfig | None = None,
) -> Scheduler:
    """Assemble a pass: the Mongo read views and Enrichment Store around `create_service`.

    Args:
        settings: Where the step runs.
        registry_client: Connection to the Asset Registry.
        scene_client: Connection to the Scene Store.
        enrichment_client: Connection to the Enrichment Store.
        enrichers: The enrichers to run, from `create_enrichers` or fakes.
        dispatcher: Policy the scene sets of a pass run under; serial when ``None``.
        config: Hyperparameters; the defaults when ``None``.
    """
    assets = MongoAssetVersionReader(registry_client, settings.registry_database)
    scene_reader = MongoSceneSetReader(scene_client, settings.scene_store_database)
    store = MongoEnrichmentStore(enrichment_client, settings.enrichment_store_database)
    return Scheduler(
        assets=assets,
        scene_reader=scene_reader,
        enrichment_reader=store,
        enricher=create_service(assets, scene_reader, store, enrichers, config),
        dispatcher=SerialDispatcher() if dispatcher is None else dispatcher,
    )


def _backoff_seconds(interval: float, failures: int) -> float:
    """Pause after ``failures`` consecutive failed passes: doubling from the interval, capped."""
    return min(interval * 2.0**failures, max(interval, _MAX_BACKOFF_SECONDS))


def run_worker(*, once: bool, sleep: Callable[[float], None] = time.sleep) -> None:
    """Entry point of the ``enrichment`` step in the worker role.

    The stores are connected first, since those checks are cheap, then the models are resolved
    and loaded once; a wrong connection string is reported before a model download starts. A
    pass that raises (a store outage, say) ends the process in ``once`` mode. In the loop it is
    logged and the next pass runs after a pause that doubles with every consecutive failure,
    from the poll interval up to a few minutes.

    Args:
        once: Run a single pass instead of looping until interrupted.
        sleep: Pause function between passes; injectable for tests.

    Raises:
        RuntimeError: When settings are missing, the loop has no poll interval, a model or
            its server is unavailable, or a store is unreachable.
    """
    settings = load_settings()
    interval = settings.poll_interval_seconds
    if not once and interval is None:
        raise RuntimeError(
            f"{settings_module.POLL_INTERVAL_SECONDS_ENV} is not set. "
            "Give the pause between passes in seconds, or run with --once."
        )
    with contextlib.ExitStack() as stack:
        registry_client = stack.enter_context(
            connect_mongo(settings.registry_uri, redacted=settings.redacted_registry_uri)
        )
        scene_client = stack.enter_context(
            connect_mongo(settings.scene_store_uri, redacted=settings.redacted_scene_store_uri)
        )
        enrichment_client = stack.enter_context(
            connect_mongo(
                settings.enrichment_store_uri, redacted=settings.redacted_enrichment_store_uri
            )
        )
        enrichers = create_enrichers(settings)
        scheduler = create_scheduler(
            settings,
            registry_client,
            scene_client,
            enrichment_client,
            enrichers=enrichers,
            dispatcher=make_dispatcher(settings.workers, stack, name="enrichment"),
        )
        logger.info(
            "enrichment fingerprints %s; %d scene set(s) at once",
            " ".join(f"{name}={fp}" for name, fp in scheduler.fingerprints.items()),
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
                logger.exception("enrichment pass failed; retrying in %.1f s", pause)
            else:
                pause = interval
                failures = 0
            sleep(pause)
