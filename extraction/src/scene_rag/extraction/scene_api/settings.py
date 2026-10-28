"""Settings of the scene-api role: where it runs and what it reads.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else. The
store variables are the ones the steps read: the role reads the stores they write, so the
names are part of the service, not the role. The role has no hyperparameters: it renders
stored data, so nothing here is fingerprinted.
"""

import dataclasses
import re
from typing import Final

ENV_PREFIX: Final = "SCENE_RAG_EXTRACTION_"
REGISTRY_URI_ENV: Final = ENV_PREFIX + "REGISTRY_URI"
REGISTRY_DATABASE_ENV: Final = ENV_PREFIX + "REGISTRY_DATABASE"
SCENE_STORE_URI_ENV: Final = ENV_PREFIX + "SCENE_STORE_URI"
SCENE_STORE_DATABASE_ENV: Final = ENV_PREFIX + "SCENE_STORE_DATABASE"
ENRICHMENT_STORE_URI_ENV: Final = ENV_PREFIX + "ENRICHMENT_STORE_URI"
ENRICHMENT_STORE_DATABASE_ENV: Final = ENV_PREFIX + "ENRICHMENT_STORE_DATABASE"
HOST_ENV: Final = ENV_PREFIX + "SCENE_API_HOST"
PORT_ENV: Final = ENV_PREFIX + "SCENE_API_PORT"
PAGE_SIZE_ENV: Final = ENV_PREFIX + "SCENE_API_PAGE_SIZE"

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8080
DEFAULT_PAGE_SIZE: Final = 100
MAX_PORT: Final = 65535

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


@dataclasses.dataclass(frozen=True)
class SceneApiSettings:
    """Deployment-specific values of the scene-api role.

    Attributes:
        registry_uri: MongoDB connection string of the Asset Registry (read only).
        registry_database: Database holding the registry collection.
        scene_store_uri: MongoDB connection string of the Scene Store (read only).
        scene_store_database: Database holding the scene set collection.
        enrichment_store_uri: MongoDB connection string of the Enrichment Store (read only).
        enrichment_store_database: Database holding the enrichment collection.
        host: Interface the server listens on.
        port: Port the server listens on; zero lets the system pick one.
        page_size: Items a page holds when a request names no limit, and the most it may ask.
    """

    registry_uri: str
    registry_database: str
    scene_store_uri: str
    scene_store_database: str
    enrichment_store_uri: str
    enrichment_store_database: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    page_size: int = DEFAULT_PAGE_SIZE

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
            ValueError: When a value cannot connect or serve.
        """
        for name in (
            "registry_uri",
            "registry_database",
            "scene_store_uri",
            "scene_store_database",
            "enrichment_store_uri",
            "enrichment_store_database",
            "host",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if not 0 <= self.port <= MAX_PORT:
            raise ValueError(f"port must be between 0 and {MAX_PORT}")
        if self.page_size < 1:
            raise ValueError("page_size must be at least 1")
