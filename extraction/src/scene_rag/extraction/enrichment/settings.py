"""Settings of the enrichment step: where and how fast it runs.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else; they
are never hashed into a fingerprint. The registry and scene store variables are the ones the
earlier steps read too: the step reads the stores they write, so the names are part of the
service, not the step.
"""

import dataclasses
import math
import re
from pathlib import Path
from typing import Final

ENV_PREFIX: Final = "SCENE_RAG_EXTRACTION_"
REGISTRY_URI_ENV: Final = ENV_PREFIX + "REGISTRY_URI"
REGISTRY_DATABASE_ENV: Final = ENV_PREFIX + "REGISTRY_DATABASE"
SCENE_STORE_URI_ENV: Final = ENV_PREFIX + "SCENE_STORE_URI"
SCENE_STORE_DATABASE_ENV: Final = ENV_PREFIX + "SCENE_STORE_DATABASE"
ENRICHMENT_STORE_URI_ENV: Final = ENV_PREFIX + "ENRICHMENT_STORE_URI"
ENRICHMENT_STORE_DATABASE_ENV: Final = ENV_PREFIX + "ENRICHMENT_STORE_DATABASE"
OLLAMA_URL_ENV: Final = ENV_PREFIX + "OLLAMA_URL"
OLLAMA_TIMEOUT_SECONDS_ENV: Final = ENV_PREFIX + "OLLAMA_TIMEOUT_SECONDS"
WHISPER_CACHE_DIR_ENV: Final = ENV_PREFIX + "WHISPER_CACHE_DIR"
WHISPER_CPU_THREADS_ENV: Final = ENV_PREFIX + "WHISPER_CPU_THREADS"
POLL_INTERVAL_SECONDS_ENV: Final = ENV_PREFIX + "POLL_INTERVAL_SECONDS"
WORKERS_ENV: Final = ENV_PREFIX + "ENRICHMENT_WORKERS"

DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS: Final = 600.0

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


@dataclasses.dataclass(frozen=True)
class EnrichmentSettings:
    """Deployment-specific values of the enrichment step.

    Attributes:
        registry_uri: MongoDB connection string of the Asset Registry (read only).
        registry_database: Database holding the registry collection.
        scene_store_uri: MongoDB connection string of the Scene Store (read only).
        scene_store_database: Database holding the scene set collection.
        enrichment_store_uri: MongoDB connection string of the Enrichment Store.
        enrichment_store_database: Database holding the enrichment collection.
        ollama_url: Base URL of the Ollama server answering the vision model.
        ollama_timeout_seconds: Longest wait for one Ollama answer; CPU inference takes
            minutes per scene.
        whisper_cache_dir: Directory the speech model is downloaded to and loaded from; the
            library's own cache when ``None``.
        whisper_cpu_threads: Threads the speech model computes on; zero leaves it to the
            library.
        poll_interval_seconds: Pause between two passes of the worker loop; ``None`` when the
            step runs a single pass and never sleeps.
        workers: Scene sets enriched at once within a pass; one means one after another. The
            model backends serialise their own work, so more mostly overlaps the vision and
            speech models with each other.
    """

    registry_uri: str
    registry_database: str
    scene_store_uri: str
    scene_store_database: str
    enrichment_store_uri: str
    enrichment_store_database: str
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    whisper_cache_dir: Path | None = None
    whisper_cpu_threads: int = 0
    poll_interval_seconds: float | None = None
    workers: int = 1

    @property
    def redacted_registry_uri(self) -> str:
        """Return `registry_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.registry_uri)

    @property
    def redacted_scene_store_uri(self) -> str:
        """Return `scene_store_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.scene_store_uri)

    @property
    def redacted_enrichment_store_uri(self) -> str:
        """Return `enrichment_store_uri` without credentials; the only loggable form."""
        return redact_uri(self.enrichment_store_uri)

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot connect or run the step.
        """
        for name in (
            "registry_uri",
            "registry_database",
            "scene_store_uri",
            "scene_store_database",
            "enrichment_store_uri",
            "enrichment_store_database",
            "ollama_url",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError("ollama_url must be an http(s) URL")
        timeout = self.ollama_timeout_seconds
        if not (math.isfinite(timeout) and timeout > 0):
            raise ValueError("ollama_timeout_seconds must be a positive finite number")
        if self.whisper_cpu_threads < 0:
            raise ValueError("whisper_cpu_threads must not be negative")
        interval = self.poll_interval_seconds
        if interval is not None and not (math.isfinite(interval) and interval > 0):
            raise ValueError("poll_interval_seconds must be a positive finite number")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
