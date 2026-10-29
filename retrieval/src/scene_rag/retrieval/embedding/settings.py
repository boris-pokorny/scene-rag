"""Settings of the embedding step: where and how fast it runs.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else; they
are never hashed into a fingerprint. The Chunk Store variables are the same as the chunking
step's: one database holds the chunks and the index ledger. The batch size is a setting, not
a hyperparameter: the vector of a text does not depend on which texts share its request.
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
BATCH_SIZE_ENV: Final = ENV_PREFIX + "EMBEDDING_BATCH_SIZE"
WORKERS_ENV: Final = ENV_PREFIX + "EMBEDDING_WORKERS"
POLL_INTERVAL_SECONDS_ENV: Final = ENV_PREFIX + "POLL_INTERVAL_SECONDS"

DEFAULT_SEARCH_INDEX_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS: Final = 120.0
DEFAULT_BATCH_SIZE: Final = 32

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


def _positive_finite(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number")


@dataclasses.dataclass(frozen=True)
class OllamaSettings:
    """Where the Ollama server is; the model-gateway stand-in.

    Attributes:
        url: Scheme, host and port of the server.
        timeout_seconds: Longest wait for one embedding answer.
    """

    url: str = DEFAULT_OLLAMA_URL
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS

    @property
    def redacted_url(self) -> str:
        """Return `url` without credentials; the only form that may be logged."""
        return redact_uri(self.url)

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot reach a server.
        """
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("ollama url must be an http(s) URL")
        _positive_finite("ollama timeout_seconds", self.timeout_seconds)


@dataclasses.dataclass(frozen=True)
class EmbeddingSettings:
    """Deployment-specific values of the embedding step.

    Attributes:
        chunk_store_uri: MongoDB connection string of the Chunk Store and the index ledger.
        chunk_store_database: Database holding the chunk and index collections.
        search_index_url: Base URL of the Search Index backend.
        ollama: Where the embedding model is served.
        search_index_timeout_seconds: Longest wait for one Search Index answer.
        batch_size: Texts embedded per request to the model.
        poll_interval_seconds: Pause between two passes of the worker loop; ``None`` when the
            step runs a single pass and never sleeps.
        workers: Chunk sets embedded at once within a pass; one means one after another.
    """

    chunk_store_uri: str
    chunk_store_database: str
    search_index_url: str
    ollama: OllamaSettings = OllamaSettings()
    search_index_timeout_seconds: float = DEFAULT_SEARCH_INDEX_TIMEOUT_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    poll_interval_seconds: float | None = None
    workers: int = 1

    @property
    def redacted_chunk_store_uri(self) -> str:
        """Return `chunk_store_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.chunk_store_uri)

    @property
    def redacted_search_index_url(self) -> str:
        """Return `search_index_url` without credentials; the only form that may be logged."""
        return redact_uri(self.search_index_url)

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot connect or run the step.
        """
        for name in ("chunk_store_uri", "chunk_store_database", "search_index_url"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if not self.search_index_url.startswith(("http://", "https://")):
            raise ValueError("search_index_url must be an http(s) URL")
        _positive_finite("search_index_timeout_seconds", self.search_index_timeout_seconds)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        interval = self.poll_interval_seconds
        if interval is not None:
            _positive_finite("poll_interval_seconds", interval)
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
