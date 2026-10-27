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
from scene_rag.extraction.ingestion.registry import Document, MongoAssetRegistry
from scene_rag.extraction.ingestion.service import IngestionService
from scene_rag.extraction.ingestion.settings import IngestionSettings
from scene_rag.extraction.ingestion.sources import FilesystemMediaSource

logger = logging.getLogger(__name__)

# Operational guardrail for the connectivity check, not a tunable.
_SERVER_SELECTION_TIMEOUT_MS: Final = 2000


class SystemClock:
    """`Clock` over the system time, truncated to the millisecond the registry stores."""

    def now(self) -> datetime:
        """Return the current UTC time at millisecond precision."""
        moment = datetime.now(UTC)
        return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def _require(environ: Mapping[str, str], name: str, hint: str) -> str:
    value = environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. {hint}")
    return value


def load_settings(environ: Mapping[str, str] | None = None) -> IngestionSettings:
    """Bind `IngestionSettings` from the environment.

    Args:
        environ: Variables to read; the process environment by default.

    Returns:
        Validated settings.

    Raises:
        RuntimeError: When a variable is missing or the settings do not validate.
    """
    env = os.environ if environ is None else environ
    poll_interval = _require(
        env, settings_module.POLL_INTERVAL_SECONDS_ENV, "Give the pause between passes in seconds."
    )
    try:
        return IngestionSettings(
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
            poll_interval_seconds=float(poll_interval),
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid settings: {exc}") from exc


def connect_mongo(
    settings: IngestionSettings, *, timeout_ms: int = _SERVER_SELECTION_TIMEOUT_MS
) -> MongoClient[Document]:
    """Connect to the registry's MongoDB and verify it answers.

    Args:
        settings: Supplies the connection string.
        timeout_ms: Server selection timeout of the connectivity check; shortened in tests.

    Raises:
        RuntimeError: When the connection string is malformed or the server cannot be
            reached.
    """
    try:
        client: MongoClient[Document] = MongoClient(
            settings.registry_uri, tz_aware=True, serverSelectionTimeoutMS=timeout_ms
        )
        client.admin.command("ping")
    except (PyMongoError, ValueError) as exc:  # pymongo raises ValueError for a bad port
        raise RuntimeError(
            f"Could not reach MongoDB at {settings.redacted_registry_uri!r}: {exc}. "
            f"Check that it is running and {settings_module.REGISTRY_URI_ENV} is correct."
        ) from exc
    logger.info("connected to MongoDB at %s", settings.redacted_registry_uri)
    return client


def create_registry(settings: IngestionSettings) -> MongoAssetRegistry:
    """Build the MongoDB Asset Registry for the configured database."""
    return MongoAssetRegistry(connect_mongo(settings), settings.registry_database)


def create_service(
    settings: IngestionSettings, config: IngestionConfig | None = None
) -> IngestionService:
    """Assemble the step: filesystem connector, MongoDB registry, system clock."""
    config = IngestionConfig() if config is None else config
    registry = create_registry(settings)
    return IngestionService(
        source=FilesystemMediaSource(
            settings.source_root, settings.source_id, settings.source_entitlements, config
        ),
        reader=registry,
        writer=registry,
        clock=SystemClock(),
        config=config,
    )


def run_worker(*, once: bool, sleep: Callable[[float], None] = time.sleep) -> None:
    """Entry point of the ``ingestion`` step in the worker role.

    A pass that raises (a registry outage, say) ends the process in ``once`` mode. In the
    loop it is logged and the next pass runs after the usual pause, so a transient outage
    does not take the worker down.

    Args:
        once: Run a single pass instead of looping until interrupted.
        sleep: Pause function between passes; injectable for tests.

    Raises:
        RuntimeError: When settings are missing or the registry is unreachable.
    """
    settings = load_settings()
    service = create_service(settings)
    logger.info("ingestion fingerprint %s", service.fingerprint)
    while True:
        if once:
            service.run_once()
            return
        try:
            service.run_once()
        except Exception:
            logger.exception("ingestion pass failed; retrying after the poll interval")
        sleep(settings.poll_interval_seconds)
