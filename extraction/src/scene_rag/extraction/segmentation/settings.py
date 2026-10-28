"""Settings of the segmentation step: where and how fast it runs.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else; they
are never hashed into a fingerprint. The registry variables are the ones ingestion reads too:
the step reads the store ingestion writes, so the names are part of the service, not the step.
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
DERIVATIVE_ROOT_ENV: Final = ENV_PREFIX + "DERIVATIVE_ROOT"
FFMPEG_BINARY_ENV: Final = ENV_PREFIX + "FFMPEG_BINARY"
POLL_INTERVAL_SECONDS_ENV: Final = ENV_PREFIX + "POLL_INTERVAL_SECONDS"

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


@dataclasses.dataclass(frozen=True)
class SegmentationSettings:
    """Deployment-specific values of the segmentation step.

    Attributes:
        registry_uri: MongoDB connection string of the Asset Registry (read only).
        registry_database: Database holding the registry collection.
        scene_store_uri: MongoDB connection string of the Scene Store.
        scene_store_database: Database holding the scene set collection.
        derivative_root: Directory the filesystem Derivative Store files keyframes and clips
            under.
        ffmpeg_binary: Name or path of the ffmpeg executable.
        poll_interval_seconds: Pause between two passes of the worker loop; ``None`` when the
            step runs a single pass and never sleeps.
    """

    registry_uri: str
    registry_database: str
    scene_store_uri: str
    scene_store_database: str
    derivative_root: Path
    ffmpeg_binary: str = "ffmpeg"
    poll_interval_seconds: float | None = None

    @property
    def redacted_registry_uri(self) -> str:
        """Return `registry_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.registry_uri)

    @property
    def redacted_scene_store_uri(self) -> str:
        """Return `scene_store_uri` without credentials; the only form that may be logged."""
        return redact_uri(self.scene_store_uri)

    def __post_init__(self) -> None:
        """Validate the values.

        The values are checked for shape only; whether ``derivative_root`` exists is a question
        for the composition root, not for a value object.

        Raises:
            ValueError: When a value cannot connect or run the step.
        """
        for name in (
            "registry_uri",
            "registry_database",
            "scene_store_uri",
            "scene_store_database",
            "ffmpeg_binary",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        interval = self.poll_interval_seconds
        if interval is not None and not (math.isfinite(interval) and interval > 0):
            raise ValueError("poll_interval_seconds must be a positive finite number")
