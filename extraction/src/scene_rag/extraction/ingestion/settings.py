"""Settings of the ingestion step: where and how fast it runs.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else; they
are never hashed into a fingerprint.
"""

import dataclasses
import math
import re
from pathlib import Path
from typing import Final

ENV_PREFIX: Final = "SCENE_RAG_EXTRACTION_"
REGISTRY_URI_ENV: Final = ENV_PREFIX + "REGISTRY_URI"
REGISTRY_DATABASE_ENV: Final = ENV_PREFIX + "REGISTRY_DATABASE"
SOURCE_ROOT_ENV: Final = ENV_PREFIX + "SOURCE_ROOT"
SOURCE_ID_ENV: Final = ENV_PREFIX + "SOURCE_ID"
SOURCE_ENTITLEMENTS_ENV: Final = ENV_PREFIX + "SOURCE_ENTITLEMENTS"
POLL_INTERVAL_SECONDS_ENV: Final = ENV_PREFIX + "POLL_INTERVAL_SECONDS"

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


@dataclasses.dataclass(frozen=True)
class IngestionSettings:
    """Deployment-specific values of the ingestion step.

    Attributes:
        registry_uri: MongoDB connection string of the Asset Registry.
        registry_database: Database holding the registry collection.
        source_root: Directory the filesystem connector lists.
        source_id: Identifier the connector stamps on its source references.
        source_entitlements: Entitlements the connector grants every item it lists; the
            filesystem has no permission model of its own.
        poll_interval_seconds: Pause between two passes of the worker loop; ``None`` when the
            step runs a single pass and never sleeps.
    """

    registry_uri: str
    registry_database: str
    source_root: Path
    source_id: str
    source_entitlements: frozenset[str]
    poll_interval_seconds: float | None = None

    @property
    def redacted_registry_uri(self) -> str:
        """Return `registry_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.registry_uri)

    def __post_init__(self) -> None:
        """Validate the values.

        The values are checked for shape only; whether ``source_root`` exists is a question
        for the composition root, not for a value object.

        Raises:
            ValueError: When a value cannot connect or run the step.
        """
        if not self.registry_uri:
            raise ValueError("registry_uri must not be empty")
        if not self.registry_database:
            raise ValueError("registry_database must not be empty")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.source_entitlements:
            raise ValueError("source_entitlements must name at least one entitlement")
        interval = self.poll_interval_seconds
        if interval is not None and not (math.isfinite(interval) and interval > 0):
            raise ValueError("poll_interval_seconds must be a positive finite number")
