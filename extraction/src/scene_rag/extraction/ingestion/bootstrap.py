"""Composition root of the ingestion step.

Binds settings from the environment and hyperparameters from `IngestionConfig`, then wires
connector, registry and clock into `IngestionService`. Nothing else reads the environment.
"""

import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.extraction.ingestion import settings as settings_module
from scene_rag.extraction.ingestion.config import IngestionConfig
from scene_rag.extraction.ingestion.models import truncate_to_millisecond
from scene_rag.extraction.ingestion.registry import Document, MongoAssetRegistry
from scene_rag.extraction.ingestion.service import IngestionService
from scene_rag.extraction.ingestion.settings import IngestionSettings
from scene_rag.extraction.ingestion.sources import FilesystemMediaSource

logger = logging.getLogger(__name__)

# Operational guardrails, not tunables: the connectivity check's patience, and the longest
# pause between retries once passes keep failing.
_SERVER_SELECTION_TIMEOUT_MS: Final = 2000
_MAX_BACKOFF_SECONDS: Final = 300.0

type ClientFactory = Callable[..., MongoClient[Document]]


class SystemClock:
    """`Clock` over the system time, truncated to the millisecond the registry stores."""

    def now(self) -> datetime:
        """Return the current UTC time at millisecond precision."""
        return truncate_to_millisecond(datetime.now(UTC))


def _require(environ: Mapping[str, str], name: str, hint: str) -> str:
    value = environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. {hint}")
    return value


def load_settings(environ: Mapping[str, str] | None = None) -> IngestionSettings:
    """Bind `IngestionSettings` from the environment.

    The poll interval is optional: a single pass never sleeps. `run_worker` demands it before
    looping.

    Args:
        environ: Variables to read; the process environment by default.

    Returns:
        Validated settings whose source root exists.

    Raises:
        RuntimeError: When a variable is missing, the settings do not validate or the source
            root is not a directory.
    """
    env = os.environ if environ is None else environ
    poll_interval = env.get(settings_module.POLL_INTERVAL_SECONDS_ENV)
    try:
        loaded = IngestionSettings(
            registry_uri=_require(
                env,
                settings_module.REGISTRY_URI_ENV,
                "Start MongoDB and point this at it, e.g. mongodb://localhost:27017.",
            ),
            registry_database=_require(
                env, settings_module.REGISTRY_DATABASE_ENV, "Name the registry database."
            ),
            source_root=Path(
                _require(
                    env,
                    settings_module.SOURCE_ROOT_ENV,
                    "Point it at the directory holding the media files.",
                )
            ).resolve(),
            source_id=_require(env, settings_module.SOURCE_ID_ENV, "Name the source."),
            source_entitlements=frozenset(
                part.strip()
                for part in _require(
                    env,
                    settings_module.SOURCE_ENTITLEMENTS_ENV,
                    "List the entitlements granted to every file, comma separated.",
                ).split(",")
                if part.strip()
            ),
            poll_interval_seconds=None if not poll_interval else float(poll_interval),
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid settings: {exc}") from exc
    if not loaded.source_root.is_dir():
        raise RuntimeError(
            f"source_root is not a directory: {loaded.source_root}. "
            f"Point {settings_module.SOURCE_ROOT_ENV} at the directory holding the media files."
        )
    return loaded


def connect_mongo(
    settings: IngestionSettings, *, client_factory: ClientFactory = MongoClient
) -> MongoClient[Document]:
    """Connect to the registry's MongoDB and verify it answers.

    Args:
        settings: Supplies the connection string.
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
            settings.registry_uri,
            tz_aware=True,
            serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
        )
        client.admin.command("ping")
    except (PyMongoError, ValueError) as exc:  # pymongo raises ValueError for a bad port
        raise RuntimeError(
            f"Could not reach MongoDB at {settings.redacted_registry_uri!r}: {exc}. "
            f"Check that it is running and {settings_module.REGISTRY_URI_ENV} is correct."
        ) from exc
    logger.info("connected to MongoDB at %s", settings.redacted_registry_uri)
    return client


def create_registry(
    client: MongoClient[Document], settings: IngestionSettings
) -> MongoAssetRegistry:
    """Build the MongoDB Asset Registry for the configured database on ``client``."""
    return MongoAssetRegistry(client, settings.registry_database)


def create_service(
    settings: IngestionSettings,
    client: MongoClient[Document],
    config: IngestionConfig | None = None,
) -> IngestionService:
    """Assemble the step: filesystem connector, MongoDB registry on ``client``, system clock."""
    config = IngestionConfig() if config is None else config
    registry = create_registry(client, settings)
    return IngestionService(
        source=FilesystemMediaSource(
            settings.source_root, settings.source_id, settings.source_entitlements, config
        ),
        reader=registry,
        writer=registry,
        clock=SystemClock(),
        config=config,
    )


def _backoff_seconds(interval: float, failures: int) -> float:
    """Pause after ``failures`` consecutive failed passes: doubling from the interval, capped."""
    return min(interval * 2.0**failures, max(interval, _MAX_BACKOFF_SECONDS))


def run_worker(*, once: bool, sleep: Callable[[float], None] = time.sleep) -> None:
    """Entry point of the ``ingestion`` step in the worker role.

    A pass that raises (a registry outage, say) ends the process in ``once`` mode. In the
    loop it is logged and the next pass runs after a pause that doubles with every
    consecutive failure, from the poll interval up to a few minutes, so a transient outage
    neither takes the worker down nor floods the log.

    Args:
        once: Run a single pass instead of looping until interrupted.
        sleep: Pause function between passes; injectable for tests.

    Raises:
        RuntimeError: When settings are missing, the loop has no poll interval or the registry
            is unreachable.
    """
    settings = load_settings()
    interval = settings.poll_interval_seconds
    if not once and interval is None:
        raise RuntimeError(
            f"{settings_module.POLL_INTERVAL_SECONDS_ENV} is not set. "
            "Give the pause between passes in seconds, or run with --once."
        )
    with connect_mongo(settings) as client:
        service = create_service(settings, client)
        logger.info("ingestion fingerprint %s", service.fingerprint)
        if once:
            service.run_once()
            return
        assert interval is not None  # checked above, before connecting
        failures = 0
        while True:
            try:
                service.run_once()
            except Exception:
                pause = _backoff_seconds(interval, failures)
                failures += 1
                logger.exception("ingestion pass failed; retrying in %.1f s", pause)
            else:
                pause = interval
                failures = 0
            sleep(pause)
