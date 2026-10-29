"""Hyperparameters of the embedding step.

The field defaults *are* the values: changing one is a code change that changes the step's
fingerprint, and so the collection the vectors land in. Nothing here comes from the
environment; connection settings, batch size and worker count live in `settings.py`. The
accepted chunking fingerprint is a hyperparameter too: a release pins it, and a new chunker
reruns embedding under a new fingerprint next to the old index.
"""

import dataclasses
import hashlib
import re
from typing import Final, Literal

FINGERPRINT_PATTERN: Final = re.compile(r"^[0-9a-f]{12}$")

type Distance = Literal["cosine", "dot", "euclid"]
"""Similarity the index is built for; the names are the step's, backends map them."""

DISTANCES: Final[frozenset[str]] = frozenset({"cosine", "dot", "euclid"})


@dataclasses.dataclass(frozen=True)
class UpstreamConfig:
    """Which chunking outputs the step accepts.

    Attributes:
        chunking_fingerprint: Only chunk sets written under this fingerprint; any when
            ``None``. Every chunk carries its own chunking fingerprint, so the index stays
            traceable either way.
    """

    chunking_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When the pin is not twelve hex digits.
        """
        fingerprint = self.chunking_fingerprint
        if fingerprint is not None and not FINGERPRINT_PATTERN.match(fingerprint):
            raise ValueError("chunking_fingerprint must be twelve hex digits or None")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Tunables of the embedding model.

    Attributes:
        name: Embedding model, as the backend names it; the fingerprint pins its digest.
        document_prefix: Put before a chunk's text when it is embedded (EmbeddingGemma's
            document prompt).
        query_prefix: Put before a query when it is embedded; recorded with the index so
            query-api embeds queries the way the index was built, never set by hand there.
        truncate: Whether the backend cuts an input beyond the context instead of failing.
        num_ctx: Context the model runs with, in tokens; where truncation happens.
    """

    name: str = "embeddinggemma"
    document_prefix: str = "title: none | text: "
    query_prefix: str = "task: search result | query: "
    truncate: bool = True
    num_ctx: int = 2048

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working model.
        """
        if not self.name:
            raise ValueError("model name must not be empty")
        if self.num_ctx < 1:
            raise ValueError("num_ctx must be at least 1")


@dataclasses.dataclass(frozen=True)
class IndexConfig:
    """Tunables of the index the vectors are written to.

    Attributes:
        distance: Similarity the collection is built for.
    """

    distance: Distance = "cosine"

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When the distance is not one the step knows.
        """
        if self.distance not in DISTANCES:
            raise ValueError(f"distance must be one of {sorted(DISTANCES)}, not {self.distance!r}")


@dataclasses.dataclass(frozen=True)
class EmbeddingConfig:
    """Every tunable of the embedding step, and nothing else.

    Attributes:
        upstream: Which chunking outputs are accepted.
        model: Tunables of the embedding model.
        index: Tunables of the index.
        hash_algorithm: ``hashlib`` algorithm that versions the records.
    """

    upstream: UpstreamConfig = UpstreamConfig()
    model: ModelConfig = ModelConfig()
    index: IndexConfig = IndexConfig()
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
