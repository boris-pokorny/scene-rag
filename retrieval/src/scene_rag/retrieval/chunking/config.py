"""Hyperparameters of the chunking step.

The field defaults *are* the values: changing one is a code change that changes the step's
fingerprint. Nothing here comes from the environment; connection settings live in
`settings.py`. The accepted upstream fingerprints are hyperparameters too: a release pins
them, and a new Media model reruns chunking under a new fingerprint next to the old chunks.
"""

import dataclasses
import hashlib
import re
from typing import Final

FINGERPRINT_PATTERN: Final = re.compile(r"^[0-9a-f]{12}$")
ENRICHER_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")


def _enricher_names(name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique: {values}")
    for value in values:
        if not ENRICHER_PATTERN.match(value):
            raise ValueError(f"{name} must be enricher names, not {value!r}")


@dataclasses.dataclass(frozen=True)
class UpstreamConfig:
    """Which Media outputs the step accepts.

    Attributes:
        segmentation_fingerprint: Only scene sets cut under this fingerprint; any when
            ``None``.
        enrichment_pins: ``<enricher>:<fingerprint>`` pairs the records must carry; an
            enricher not pinned here is taken at whichever fingerprint is complete over the
            scene set (the greatest when several are).
        required_enrichers: Enrichers a scene set must be completely covered by before it is
            chunked; the records fetched carry exactly these, one fingerprint each.
    """

    segmentation_fingerprint: str | None = None
    enrichment_pins: tuple[str, ...] = ()
    required_enrichers: tuple[str, ...] = ("description", "transcript")

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot select scene records.
        """
        fingerprint = self.segmentation_fingerprint
        if fingerprint is not None and not FINGERPRINT_PATTERN.match(fingerprint):
            raise ValueError("segmentation_fingerprint must be twelve hex digits or None")
        _enricher_names("required_enrichers", self.required_enrichers)
        if not self.required_enrichers:
            raise ValueError("required_enrichers must name at least one enricher")
        pinned: list[str] = []
        for pin in self.enrichment_pins:
            enricher, colon, digest = pin.partition(":")
            if not colon or not ENRICHER_PATTERN.match(enricher):
                raise ValueError(f"enrichment_pins must be <enricher>:<fingerprint>, not {pin!r}")
            if not FINGERPRINT_PATTERN.match(digest):
                raise ValueError(f"enrichment_pins fingerprint must be twelve hex digits: {pin!r}")
            if enricher not in self.required_enrichers:
                raise ValueError(f"enrichment_pins names {enricher!r}, which is not required")
            pinned.append(enricher)
        if len(set(pinned)) != len(pinned):
            raise ValueError(f"enrichment_pins must pin each enricher once: {self.enrichment_pins}")


@dataclasses.dataclass(frozen=True)
class PerSceneConfig:
    """Tunables of the per-scene chunker.

    Attributes:
        enrichers: Enrichers whose text goes into a chunk, in section order; others are
            ignored.
        section_separator: Put between the sections of different enrichers.
        segment_separator: Put between the segments of one enricher.
    """

    enrichers: tuple[str, ...] = ("description", "transcript")
    section_separator: str = "\n\n"
    segment_separator: str = " "

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working chunker.
        """
        _enricher_names("enrichers", self.enrichers)
        if not self.enrichers:
            raise ValueError("enrichers must name at least one enricher")


@dataclasses.dataclass(frozen=True)
class ChunkingConfig:
    """Every tunable of the chunking step, and nothing else.

    Attributes:
        upstream: Which Media outputs are accepted.
        per_scene: Tunables of the per-scene chunker.
        hash_algorithm: ``hashlib`` algorithm that versions the records.
    """

    upstream: UpstreamConfig = UpstreamConfig()
    per_scene: PerSceneConfig = PerSceneConfig()
    hash_algorithm: str = "sha256"

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working step.
        """
        if self.hash_algorithm not in hashlib.algorithms_available:
            raise ValueError(f"unknown hash algorithm: {self.hash_algorithm!r}")
        if hashlib.new(self.hash_algorithm).digest_size == 0:  # shake: no fixed-length digest
            raise ValueError(
                f"hash algorithm must have a fixed digest size: {self.hash_algorithm!r}"
            )
        missing = set(self.per_scene.enrichers) - set(self.upstream.required_enrichers)
        if missing:
            raise ValueError(
                f"per_scene.enrichers names {sorted(missing)}, which upstream does not require"
            )
