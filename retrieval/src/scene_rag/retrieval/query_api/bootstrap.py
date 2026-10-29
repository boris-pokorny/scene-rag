"""Composition root of the query-api role.

Binds settings from the environment and hyperparameters from `QueryConfig`, connects the
index ledger, the Search Index, Ollama and the scene-api, selects and verifies the index to
serve, resolves the generator, wires every policy into `QueryServiceImpl`, builds the
application and the MCP server over it, and serves them. Nothing else reads the
environment, and nothing else names a backend, a model or a server.
"""

import contextlib
import logging
import os
from collections.abc import Callable, Mapping
from typing import Final, Literal

import uvicorn
from fastapi import FastAPI
from mcp.server.mcpserver import MCPServer
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.retrieval.query_api import models, ollama_pull
from scene_rag.retrieval.query_api import settings as settings_module
from scene_rag.retrieval.query_api.app import create_app
from scene_rag.retrieval.query_api.caller import bind_caller, reset_caller
from scene_rag.retrieval.query_api.config import QueryConfig
from scene_rag.retrieval.query_api.fingerprint import (
    FINGERPRINTED_MODULES,
    compute_query_config_id,
)
from scene_rag.retrieval.query_api.http_transport import Transport, UrllibTransport
from scene_rag.retrieval.query_api.index_ledger import Document, MongoIndexLedger
from scene_rag.retrieval.query_api.mcp_server import create_mcp_server
from scene_rag.retrieval.query_api.ollama import (
    OllamaAnswerGenerator,
    OllamaClient,
    OllamaQueryEmbedder,
)
from scene_rag.retrieval.query_api.prompt import PromptBuilder
from scene_rag.retrieval.query_api.protocols import (
    AnswerGenerator,
    IndexLedgerReader,
    IndexUnavailableError,
    ModelError,
    QueryEmbedder,
    SceneApiError,
    SceneReader,
    SearchIndexReader,
)
from scene_rag.retrieval.query_api.refusal import ScoreThresholdPolicy
from scene_rag.retrieval.query_api.reranker import NoReranker
from scene_rag.retrieval.query_api.retriever import DenseRetriever
from scene_rag.retrieval.query_api.scene_api_client import SceneApiClient
from scene_rag.retrieval.query_api.search_index import QdrantSearchReader
from scene_rag.retrieval.query_api.service import QueryServiceImpl
from scene_rag.retrieval.query_api.settings import OllamaSettings, QuerySettings

logger = logging.getLogger(__name__)

# An operational guardrail, not a tunable: the connectivity check's patience.
_SERVER_SELECTION_TIMEOUT_MS: Final = 2000

type ClientFactory = Callable[..., MongoClient[Document]]
type Server = Callable[[FastAPI, str, int], None]
"""Serves an application on a host and port until told to stop."""
type StdioRunner = Callable[[MCPServer], None]
"""Serves an MCP server over the process's standard streams until they close."""
type TransportName = Literal["http", "stdio"]


class NoRetiredFilter:
    """`RetiredFilter` that blocks nothing: retirement events are not consumed yet."""

    def is_retired(self, asset_version: str) -> bool:
        """Return ``False``."""
        return False


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


def _or_default(value: float | None, default: float) -> float:
    return default if value is None else value


def _names(environ: Mapping[str, str], name: str) -> tuple[str, ...] | None:
    """Read an optional comma-separated list, trimmed, empties dropped; ``None`` when unset."""
    value = environ.get(name)
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_ollama_settings(environ: Mapping[str, str] | None = None) -> OllamaSettings:
    """Bind `OllamaSettings` from the environment; every variable has a default.

    Raises:
        RuntimeError: When a value does not validate.
    """
    env = os.environ if environ is None else environ
    try:
        return OllamaSettings(
            url=env.get(settings_module.OLLAMA_URL_ENV) or settings_module.DEFAULT_OLLAMA_URL,
            timeout_seconds=_or_default(
                _number(env, settings_module.OLLAMA_TIMEOUT_SECONDS_ENV),
                settings_module.DEFAULT_OLLAMA_TIMEOUT_SECONDS,
            ),
            generator_timeout_seconds=_or_default(
                _number(env, settings_module.GENERATOR_TIMEOUT_SECONDS_ENV),
                settings_module.DEFAULT_GENERATOR_TIMEOUT_SECONDS,
            ),
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid settings: {exc}") from exc


def load_settings(environ: Mapping[str, str] | None = None) -> QuerySettings:
    """Bind `QuerySettings` from the environment.

    The store, index and scene-api variables are required; timeouts, host, port and the
    allowed hosts have defaults.

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
        port = _count(env, settings_module.PORT_ENV, minimum=0)
        allowed_hosts = _names(env, settings_module.ALLOWED_HOSTS_ENV)
        return QuerySettings(
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
            scene_api_url=_require(
                env,
                settings_module.SCENE_API_URL_ENV,
                "Start the scene-api of extraction and point this at it, "
                "e.g. http://127.0.0.1:8080.",
            ),
            ollama=ollama,
            search_index_timeout_seconds=_or_default(
                _number(env, settings_module.SEARCH_INDEX_TIMEOUT_SECONDS_ENV),
                settings_module.DEFAULT_SEARCH_INDEX_TIMEOUT_SECONDS,
            ),
            scene_api_timeout_seconds=_or_default(
                _number(env, settings_module.SCENE_API_TIMEOUT_SECONDS_ENV),
                settings_module.DEFAULT_SCENE_API_TIMEOUT_SECONDS,
            ),
            host=env.get(settings_module.HOST_ENV) or settings_module.DEFAULT_HOST,
            port=settings_module.DEFAULT_PORT if port is None else port,
            allowed_hosts=(
                settings_module.DEFAULT_ALLOWED_HOSTS if allowed_hosts is None else allowed_hosts
            ),
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


def connect_ollama(settings: OllamaSettings, transport: Transport | None = None) -> OllamaClient:
    """Build the Ollama client and check that the server answers.

    Args:
        settings: Where the server is and how long to wait for it.
        transport: The HTTP layer; the standard library outside tests.

    Raises:
        RuntimeError: When the server does not answer.
    """
    client = OllamaClient(
        settings.url,
        UrllibTransport() if transport is None else transport,
        embed_timeout_seconds=settings.timeout_seconds,
        chat_timeout_seconds=settings.generator_timeout_seconds,
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
    settings: QuerySettings, transport: Transport | None = None
) -> SearchIndexReader:
    """Build the Search Index reader and check that the backend answers.

    The registration point of the backend: another vector database is composed here, never
    selected by a flag elsewhere.

    Args:
        settings: Where the backend is and how long to wait for it.
        transport: The HTTP layer; the standard library outside tests.

    Raises:
        RuntimeError: When the backend does not answer.
    """
    reader = QdrantSearchReader(
        settings.search_index_url,
        UrllibTransport() if transport is None else transport,
        settings.search_index_timeout_seconds,
    )
    try:
        version = reader.version()
    except IndexUnavailableError as exc:
        raise RuntimeError(
            f"The Search Index is not reachable at {settings.redacted_search_index_url}: {exc}. "
            "Start Qdrant (`docker compose up -d qdrant`) or point "
            f"{settings_module.SEARCH_INDEX_URL_ENV} at it."
        ) from exc
    logger.info("connected to Qdrant %s at %s", version, settings.redacted_search_index_url)
    return reader


def connect_scene_api(settings: QuerySettings, transport: Transport | None = None) -> SceneReader:
    """Build the scene-api client and check that the API answers.

    Args:
        settings: Where the API is and how long to wait for it.
        transport: The HTTP layer; the standard library outside tests.

    Raises:
        RuntimeError: When the API does not answer.
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
            "Start it (`docker compose up -d scene-api`) or point "
            f"{settings_module.SCENE_API_URL_ENV} at it."
        ) from exc
    logger.info("connected to scene-api at %s", settings.redacted_scene_api_url)
    return client


def select_index(ledger: IndexLedgerReader, config: QueryConfig) -> models.IndexView:
    """Return the index the configuration pins, or the newest one when it pins none.

    Raises:
        RuntimeError: When the pinned index does not exist, or the ledger is empty.
    """
    try:
        pinned = config.upstream.embedding_fingerprint
        if pinned is not None:
            index = ledger.get_index(pinned)
            if index is None:
                raise RuntimeError(
                    f"No index under embedding fingerprint {pinned}, which the query config "
                    "pins. Run the embedding worker under that fingerprint, or pin another."
                )
            logger.info("serving the pinned index %s", pinned)
            return index
        index = ledger.newest_index()
    except IndexUnavailableError as exc:
        raise RuntimeError(f"The index ledger cannot be read: {exc}") from exc
    if index is None:
        raise RuntimeError(
            "The index ledger is empty: nothing to serve. Run the embedding worker first."
        )
    logger.info("serving the newest index %s, prepared %s", index.fingerprint, index.created_at)
    return index


def check_index(
    ollama: OllamaClient, search_index: SearchIndexReader, index: models.IndexView
) -> QueryEmbedder:
    """Verify the index can be served, and return the embedder that embeds queries for it.

    The model Ollama holds under the index's model name must have the digest the index was
    built with, the collection must exist with the index's dimensions, and a probe query
    must embed to that many dimensions.

    Raises:
        RuntimeError: When any of that does not hold.
    """
    try:
        model = ollama.resolve_model(index.model.name)
    except ModelError as exc:
        raise RuntimeError(f"{exc} Or run the pull-models role.") from exc
    if model.digest != index.model.digest:
        raise RuntimeError(
            f"Ollama holds {index.model.name} as {model.digest}, the index {index.fingerprint} "
            f"was built with {index.model.digest}. Pull the pinned model or rebuild the index."
        )
    try:
        info = search_index.describe(index.collection)
    except IndexUnavailableError as exc:
        raise RuntimeError(f"Could not inspect the collection {index.collection}: {exc}") from exc
    if info is None:
        raise RuntimeError(
            f"The collection {index.collection} of index {index.fingerprint} does not exist."
        )
    if info.dimensions != index.dimensions:
        raise RuntimeError(
            f"The collection {index.collection} holds {info.dimensions}-dimensional vectors, "
            f"the index record says {index.dimensions}."
        )
    embedder = OllamaQueryEmbedder(ollama, index)
    try:
        embedder.embed("dimension probe")
    except ModelError as exc:
        raise RuntimeError(f"Could not embed a probe query: {exc}") from exc
    logger.info(
        "index %s: %s %s, %d dimensions, %d point(s)",
        index.fingerprint,
        index.model.name,
        index.model.digest,
        index.dimensions,
        info.point_count,
    )
    return embedder


def create_generator(ollama: OllamaClient, config: QueryConfig) -> AnswerGenerator:
    """Return the answer generator in force: the configured model on Ollama, digest resolved.

    The registration point of the `AnswerGenerator` extension point: another gateway is
    composed here, never selected by a flag elsewhere.

    Raises:
        RuntimeError: When the server does not hold the model.
    """
    try:
        model = ollama.resolve_model(config.generator.name)
    except ModelError as exc:
        raise RuntimeError(f"{exc} Or run the pull-models role.") from exc
    return OllamaAnswerGenerator(ollama, model, config.generator)


def create_service(
    *,
    embedder: QueryEmbedder,
    search_index: SearchIndexReader,
    generator: AnswerGenerator,
    scenes: SceneReader,
    index: models.IndexView,
    config: QueryConfig,
) -> QueryServiceImpl:
    """Assemble the use cases: every policy the configuration selects, over the backends.

    The ``query_config`` id is computed here, from the configuration, the output-defining
    modules and the generator's digest.
    """
    served = models.Served(
        query_config=compute_query_config_id(config, FINGERPRINTED_MODULES, generator.model.digest),
        embedding_fingerprint=index.fingerprint,
        chunking_fingerprint=index.chunking_fingerprint,
        collection=index.collection,
    )
    return QueryServiceImpl(
        retriever=DenseRetriever(embedder, search_index, index),
        reranker=NoReranker(),
        refusal=ScoreThresholdPolicy(config.refusal),
        prompt=PromptBuilder(config.prompt),
        generator=generator,
        scenes=scenes,
        retired=NoRetiredFilter(),
        config=config,
        served=served,
    )


def serve_with_uvicorn(app: FastAPI, host: str, port: int) -> None:
    """Serve ``app`` until interrupted; uvicorn handles the signals.

    The service's logging configuration stays in force: uvicorn is told not to install its
    own.
    """
    uvicorn.run(app, host=host, port=port, log_config=None)


def run_stdio(server: MCPServer) -> None:
    """Serve the MCP server over standard input and output until they close."""
    server.run("stdio")


def run_server(
    *,
    transport: TransportName = "http",
    entitlements: frozenset[str] = frozenset(),
    serve: Server = serve_with_uvicorn,
    stdio: StdioRunner = run_stdio,
) -> None:
    """Entry point of the ``query-api`` role.

    The backends are connected and the index verified first, so a wrong connection string,
    a missing model or an index built with another model is reported before anything binds.
    Over HTTP the application serves the REST routes and the MCP transport on one port; over
    stdio the MCP server alone speaks on the process's standard streams, as the caller the
    command line named.

    Args:
        transport: ``http`` (the server) or ``stdio`` (the MCP server for a local client).
        entitlements: The caller's entitlements over stdio, where no gateway asserts them.
        serve: Runs the application on the host and port; uvicorn outside tests.
        stdio: Runs the MCP server over the standard streams; the SDK outside tests.

    Raises:
        RuntimeError: When settings are missing, a backend is unreachable, no index can be
            served or a model is missing.
    """
    settings = load_settings()
    # One config for the whole process: the id is computed under the values the service
    # answers with.
    config = QueryConfig()
    with contextlib.ExitStack() as stack:
        mongo = stack.enter_context(
            connect_mongo(settings.chunk_store_uri, redacted=settings.redacted_chunk_store_uri)
        )
        ollama = connect_ollama(settings.ollama)
        search_index = connect_search_index(settings)
        scenes = connect_scene_api(settings)
        index = select_index(MongoIndexLedger(mongo, settings.chunk_store_database), config)
        embedder = check_index(ollama, search_index, index)
        generator = create_generator(ollama, config)
        service = create_service(
            embedder=embedder,
            search_index=search_index,
            generator=generator,
            scenes=scenes,
            index=index,
            config=config,
        )
        logger.info(
            "query config %s: generator %s %s; serving index %s from %s",
            service.served.query_config,
            generator.model.name,
            generator.model.digest,
            index.fingerprint,
            index.collection,
        )
        mcp_server = create_mcp_server(service)
        if transport == "stdio":
            token = bind_caller(models.Caller(entitlements))
            logger.info("query-api serving MCP over stdio as %s", sorted(entitlements))
            try:
                stdio(mcp_server)
            finally:
                reset_caller(token)
            return
        logger.info("query-api serving on http://%s:%d", settings.host, settings.port)
        app = create_app(service, mcp_server, allowed_hosts=settings.allowed_hosts)
        serve(app, settings.host, settings.port)


def pull_models(
    *,
    environ: Mapping[str, str] | None = None,
    config: QueryConfig | None = None,
    transport: Transport | None = None,
) -> None:
    """The ``pull-models`` role's share: put the generator model on the Ollama server.

    The model is the one `QueryConfig` names, so a deployment pulls what the role will
    resolve, without naming it a second time. A model the server already holds is not pulled
    again.

    Args:
        environ: Variables to read; the process environment by default.
        config: Hyperparameters; the defaults when ``None``.
        transport: HTTP layer for Ollama; the standard library outside tests.

    Raises:
        RuntimeError: When the settings do not validate, Ollama is unreachable or the pull
            fails.
    """
    settings = load_ollama_settings(environ)
    config = QueryConfig() if config is None else config
    name = config.generator.name
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
