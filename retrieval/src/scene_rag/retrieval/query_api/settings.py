"""Settings of the query-api role: where it runs and what it reads.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else; they
are never hashed into the ``query_config`` id. The store, index, model and scene-api
variables are the ones the steps read: the role reads what they wrote, so the names are part
of the service, not the role.
"""

import dataclasses
import math
import re
from typing import Final

ENV_PREFIX: Final = "SCENE_RAG_RETRIEVAL_"
CHUNK_STORE_URI_ENV: Final = ENV_PREFIX + "CHUNK_STORE_URI"
CHUNK_STORE_DATABASE_ENV: Final = ENV_PREFIX + "CHUNK_STORE_DATABASE"
SEARCH_INDEX_URL_ENV: Final = ENV_PREFIX + "SEARCH_INDEX_URL"
SEARCH_INDEX_TIMEOUT_SECONDS_ENV: Final = ENV_PREFIX + "SEARCH_INDEX_TIMEOUT_SECONDS"
OLLAMA_URL_ENV: Final = ENV_PREFIX + "OLLAMA_URL"
OLLAMA_TIMEOUT_SECONDS_ENV: Final = ENV_PREFIX + "OLLAMA_TIMEOUT_SECONDS"
SCENE_API_URL_ENV: Final = ENV_PREFIX + "SCENE_API_URL"
SCENE_API_TIMEOUT_SECONDS_ENV: Final = ENV_PREFIX + "SCENE_API_TIMEOUT_SECONDS"
HOST_ENV: Final = ENV_PREFIX + "QUERY_API_HOST"
PORT_ENV: Final = ENV_PREFIX + "QUERY_API_PORT"
ALLOWED_HOSTS_ENV: Final = ENV_PREFIX + "QUERY_API_ALLOWED_HOSTS"
GENERATOR_TIMEOUT_SECONDS_ENV: Final = ENV_PREFIX + "QUERY_API_GENERATOR_TIMEOUT_SECONDS"

DEFAULT_SEARCH_INDEX_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS: Final = 120.0
DEFAULT_SCENE_API_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_GENERATOR_TIMEOUT_SECONDS: Final = 300.0
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8081
DEFAULT_ALLOWED_HOSTS: Final = ("127.0.0.1", "localhost")
MAX_PORT: Final = 65535

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


def _positive_finite(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number")


def _http_url(name: str, value: str) -> None:
    if not value.startswith(("http://", "https://")):
        raise ValueError(f"{name} must be an http(s) URL")


@dataclasses.dataclass(frozen=True)
class OllamaSettings:
    """Where the Ollama server is; the model-gateway stand-in.

    Attributes:
        url: Scheme, host and port of the server.
        timeout_seconds: Longest wait for one embedding answer.
        generator_timeout_seconds: Longest wait for one generated answer.
    """

    url: str = DEFAULT_OLLAMA_URL
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    generator_timeout_seconds: float = DEFAULT_GENERATOR_TIMEOUT_SECONDS

    @property
    def redacted_url(self) -> str:
        """Return `url` without credentials; the only form that may be logged."""
        return redact_uri(self.url)

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot reach a server.
        """
        _http_url("ollama url", self.url)
        _positive_finite("ollama timeout_seconds", self.timeout_seconds)
        _positive_finite("ollama generator_timeout_seconds", self.generator_timeout_seconds)


@dataclasses.dataclass(frozen=True)
class QuerySettings:
    """Deployment-specific values of the query-api role.

    Attributes:
        chunk_store_uri: MongoDB connection string of the index ledger (read only).
        chunk_store_database: Database holding the ledger collections.
        search_index_url: Base URL of the Search Index backend.
        scene_api_url: Base URL of the scene-api of the extraction service.
        ollama: Where the models are served.
        search_index_timeout_seconds: Longest wait for one Search Index answer.
        scene_api_timeout_seconds: Longest wait for one scene-api answer.
        host: Interface the server listens on.
        port: Port the server listens on; zero lets the system pick one.
        allowed_hosts: Host header values served, without the port; any other is refused,
            so a page whose name was rebound to this address cannot send the trusted
            entitlements header. ``*`` serves every host, for a gateway that checks it.
    """

    chunk_store_uri: str
    chunk_store_database: str
    search_index_url: str
    scene_api_url: str
    ollama: OllamaSettings = OllamaSettings()
    search_index_timeout_seconds: float = DEFAULT_SEARCH_INDEX_TIMEOUT_SECONDS
    scene_api_timeout_seconds: float = DEFAULT_SCENE_API_TIMEOUT_SECONDS
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS

    @property
    def redacted_chunk_store_uri(self) -> str:
        """Return `chunk_store_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.chunk_store_uri)

    @property
    def redacted_search_index_url(self) -> str:
        """Return `search_index_url` without credentials; the only form that may be logged."""
        return redact_uri(self.search_index_url)

    @property
    def redacted_scene_api_url(self) -> str:
        """Return `scene_api_url` without credentials; the only form that may be logged."""
        return redact_uri(self.scene_api_url)

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot connect or serve.
        """
        for name in ("chunk_store_uri", "chunk_store_database", "host"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        _http_url("search_index_url", self.search_index_url)
        _http_url("scene_api_url", self.scene_api_url)
        _positive_finite("search_index_timeout_seconds", self.search_index_timeout_seconds)
        _positive_finite("scene_api_timeout_seconds", self.scene_api_timeout_seconds)
        if not 0 <= self.port <= MAX_PORT:
            raise ValueError(f"port must be between 0 and {MAX_PORT}")
        if not self.allowed_hosts or not all(self.allowed_hosts):
            raise ValueError("allowed_hosts must name at least one host, and no empty one")
