"""Settings of the chunking step: where and how fast it runs.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else; they
are never hashed into a fingerprint. The scene-api is reached over HTTP: this service never
opens a Media store.
"""

import dataclasses
import math
import re
from typing import Final

ENV_PREFIX: Final = "SCENE_RAG_RETRIEVAL_"
SCENE_API_URL_ENV: Final = ENV_PREFIX + "SCENE_API_URL"
SCENE_API_TIMEOUT_SECONDS_ENV: Final = ENV_PREFIX + "SCENE_API_TIMEOUT_SECONDS"
CHUNK_STORE_URI_ENV: Final = ENV_PREFIX + "CHUNK_STORE_URI"
CHUNK_STORE_DATABASE_ENV: Final = ENV_PREFIX + "CHUNK_STORE_DATABASE"
POLL_INTERVAL_SECONDS_ENV: Final = ENV_PREFIX + "POLL_INTERVAL_SECONDS"
WORKERS_ENV: Final = ENV_PREFIX + "CHUNKING_WORKERS"

DEFAULT_SCENE_API_TIMEOUT_SECONDS: Final = 30.0

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


@dataclasses.dataclass(frozen=True)
class ChunkingSettings:
    """Deployment-specific values of the chunking step.

    Attributes:
        scene_api_url: Base URL of the extraction service's scene-api.
        scene_api_timeout_seconds: Longest wait for one scene-api answer.
        chunk_store_uri: MongoDB connection string of the Chunk Store.
        chunk_store_database: Database holding the chunk collections.
        poll_interval_seconds: Pause between two passes of the worker loop; ``None`` when the
            step runs a single pass and never sleeps.
        workers: Scene sets chunked at once within a pass; one means one after another.
    """

    scene_api_url: str
    chunk_store_uri: str
    chunk_store_database: str
    scene_api_timeout_seconds: float = DEFAULT_SCENE_API_TIMEOUT_SECONDS
    poll_interval_seconds: float | None = None
    workers: int = 1

    @property
    def redacted_chunk_store_uri(self) -> str:
        """Return `chunk_store_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.chunk_store_uri)

    @property
    def redacted_scene_api_url(self) -> str:
        """Return `scene_api_url` without credentials; the only form that may be logged."""
        return redact_uri(self.scene_api_url)

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot connect or run the step.
        """
        for name in ("scene_api_url", "chunk_store_uri", "chunk_store_database"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if not self.scene_api_url.startswith(("http://", "https://")):
            raise ValueError("scene_api_url must be an http(s) URL")
        timeout = self.scene_api_timeout_seconds
        if not (math.isfinite(timeout) and timeout > 0):
            raise ValueError("scene_api_timeout_seconds must be a positive finite number")
        interval = self.poll_interval_seconds
        if interval is not None and not (math.isfinite(interval) and interval > 0):
            raise ValueError("poll_interval_seconds must be a positive finite number")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
