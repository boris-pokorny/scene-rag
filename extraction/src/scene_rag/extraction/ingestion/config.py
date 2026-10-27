"""Hyperparameters of the ingestion step.

The field defaults *are* the values: changing one is a code change that changes the step's
fingerprint. Nothing here comes from the environment; connection settings live in
`settings.py`.
"""

import dataclasses
import hashlib
import math
from collections.abc import Mapping
from types import MappingProxyType


def _default_media_types() -> Mapping[str, str]:
    return MappingProxyType(
        {
            ".mp4": "video/mp4",
            ".m4v": "video/x-m4v",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
        }
    )


@dataclasses.dataclass(frozen=True)
class IngestionConfig:
    """Every tunable of the ingestion step, and nothing else.

    Attributes:
        media_types: Accepted file extensions (lowercase, with the leading dot) mapped to the
            MIME type recorded for them. Items with other extensions are not assets.
        recursive: Whether connectors over hierarchical sources descend into subdirectories.
        hash_algorithm: ``hashlib`` algorithm that defines an asset version.
        min_size_bytes: Smallest item accepted as an asset; smaller ones are quarantined.
        min_age_seconds: Items modified more recently than this are not hashed in the current
            pass but deferred to the next one, so a file still being copied in is not
            registered under a digest of partial content.
    """

    media_types: Mapping[str, str] = dataclasses.field(default_factory=_default_media_types)
    recursive: bool = True
    hash_algorithm: str = "sha256"
    min_size_bytes: int = 1
    min_age_seconds: float = 5.0

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working step.
        """
        if not self.media_types:
            raise ValueError("media_types must accept at least one extension")
        for extension, media_type in self.media_types.items():
            if not extension.startswith(".") or extension != extension.lower():
                raise ValueError(f"extension must be lowercase with a leading dot: {extension!r}")
            if "/" not in media_type:
                raise ValueError(f"not a MIME type: {media_type!r} for {extension!r}")
        if self.hash_algorithm not in hashlib.algorithms_available:
            raise ValueError(f"unknown hash algorithm: {self.hash_algorithm!r}")
        if self.min_size_bytes < 0:
            raise ValueError("min_size_bytes must not be negative")
        if not (math.isfinite(self.min_age_seconds) and self.min_age_seconds >= 0):
            raise ValueError("min_age_seconds must be a non-negative finite number")
