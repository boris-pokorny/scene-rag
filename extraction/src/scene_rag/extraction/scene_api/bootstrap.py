"""Composition root of the scene-api role.

Binds settings from the environment, connects the stores, wires the read views into
`SceneCatalogService`, builds the application over it and serves it. Nothing else reads the
environment, and nothing else names the server.
"""

import contextlib
import logging
import os
from collections.abc import Callable, Mapping
from typing import Final

import uvicorn
from fastapi import FastAPI
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.extraction.scene_api import settings as settings_module
from scene_rag.extraction.scene_api.app import create_app
from scene_rag.extraction.scene_api.asset_registry import Document, MongoAssetVersionReader
from scene_rag.extraction.scene_api.derivative_reader import FileDerivativeReader
from scene_rag.extraction.scene_api.enrichment_store import MongoEnrichmentReader
from scene_rag.extraction.scene_api.protocols import (
    AssetVersionReader,
    DerivativeReader,
    EnrichmentReader,
    SceneSetReader,
)
from scene_rag.extraction.scene_api.scene_store import MongoSceneSetReader
from scene_rag.extraction.scene_api.service import SceneCatalogService
from scene_rag.extraction.scene_api.settings import SceneApiSettings

logger = logging.getLogger(__name__)

# Operational guardrails, not tunables: the connectivity check's patience, and how many scans
# of the Scene Store one listing request may make to fill its page (see `SceneCatalogService`).
_SERVER_SELECTION_TIMEOUT_MS: Final = 2000
PAGE_FILL: Final = 4

type ClientFactory = Callable[..., MongoClient[Document]]
type Server = Callable[[FastAPI, str, int], None]
"""Serves an application on a host and port until told to stop."""


def _require(environ: Mapping[str, str], name: str, hint: str) -> str:
    value = environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. {hint}")
    return value


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


def load_settings(environ: Mapping[str, str] | None = None) -> SceneApiSettings:
    """Bind `SceneApiSettings` from the environment.

    The store variables are required; host, port and page size have defaults.

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
        port = _count(env, settings_module.PORT_ENV, minimum=0)
        page_size = _count(env, settings_module.PAGE_SIZE_ENV, minimum=1)
        return SceneApiSettings(
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
            host=env.get(settings_module.HOST_ENV) or settings_module.DEFAULT_HOST,
            port=settings_module.DEFAULT_PORT if port is None else port,
            page_size=settings_module.DEFAULT_PAGE_SIZE if page_size is None else page_size,
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


def create_service(
    assets: AssetVersionReader,
    scene_sets: SceneSetReader,
    enrichments: EnrichmentReader,
    derivatives: DerivativeReader,
    *,
    page_size: int,
    page_fill: int = PAGE_FILL,
) -> SceneCatalogService:
    """Assemble the use cases over the given read views.

    Args:
        assets: Read view of the Asset Registry.
        scene_sets: Read view of the Scene Store.
        enrichments: Read view of the Enrichment Store.
        derivatives: Reader of the bytes the scene sets reference.
        page_size: Default and ceiling of a page.
        page_fill: Most scans one listing request makes to fill its page.
    """
    return SceneCatalogService(
        assets=assets,
        scene_sets=scene_sets,
        enrichments=enrichments,
        derivatives=derivatives,
        page_size=page_size,
        page_fill=page_fill,
    )


def create_catalog(
    settings: SceneApiSettings,
    registry_client: MongoClient[Document],
    scene_client: MongoClient[Document],
    enrichment_client: MongoClient[Document],
) -> SceneCatalogService:
    """Assemble the use cases over the Mongo read views and the filesystem derivatives.

    Args:
        settings: Where the stores are and how large a page is.
        registry_client: Connection to the Asset Registry.
        scene_client: Connection to the Scene Store.
        enrichment_client: Connection to the Enrichment Store.
    """
    return create_service(
        MongoAssetVersionReader(registry_client, settings.registry_database),
        MongoSceneSetReader(scene_client, settings.scene_store_database),
        MongoEnrichmentReader(enrichment_client, settings.enrichment_store_database),
        FileDerivativeReader(),
        page_size=settings.page_size,
    )


def serve_with_uvicorn(app: FastAPI, host: str, port: int) -> None:
    """Serve ``app`` until interrupted; uvicorn handles the signals.

    The service's logging configuration stays in force: uvicorn is told not to install its
    own.
    """
    uvicorn.run(app, host=host, port=port, log_config=None)


def run_server(*, serve: Server = serve_with_uvicorn) -> None:
    """Entry point of the ``scene-api`` role.

    The stores are connected first, so a wrong connection string is reported before the
    server binds. The clients stay open while the server runs and close when it returns.

    Args:
        serve: Runs the application on the host and port; uvicorn outside tests.

    Raises:
        RuntimeError: When settings are missing or a store is unreachable.
    """
    settings = load_settings()
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
        catalog = create_catalog(settings, registry_client, scene_client, enrichment_client)
        logger.info(
            "scene-api serving on http://%s:%d, pages of %d",
            settings.host,
            settings.port,
            settings.page_size,
        )
        serve(create_app(catalog), settings.host, settings.port)
