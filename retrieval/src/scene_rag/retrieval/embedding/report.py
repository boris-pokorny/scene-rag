"""What an embedding pass reports: the outcome of each chunk set and the pass's counters.

Orchestration vocabulary, deliberately outside the fingerprint.
"""

import dataclasses
from collections import Counter
from collections.abc import Iterable
from typing import Literal

type Outcome = Literal["indexed", "duplicate", "deferred", "failed"]
"""What became of one chunk set in a pass; the field names of `EmbeddingReport`."""


@dataclasses.dataclass(frozen=True)
class EmbeddingReport:
    """Counters of one embedding pass, per chunk set.

    Attributes:
        indexed: Chunk sets whose index set was written.
        duplicate: Chunk sets whose index set another worker wrote first; nothing changed.
        deferred: Chunk sets whose chunks were not all there at fetch time; left pending.
        failed: Chunk sets that raised; the pass continued past them.
    """

    indexed: int = 0
    duplicate: int = 0
    deferred: int = 0
    failed: int = 0

    @classmethod
    def counting(cls, outcomes: Iterable[Outcome]) -> EmbeddingReport:
        """Tally the outcomes of a pass, in whatever order they arrive."""
        counts: Counter[str] = Counter(outcomes)
        return cls(**{field.name: counts[field.name] for field in dataclasses.fields(cls)})

    def __str__(self) -> str:
        """Render the counters as ``name=value`` pairs for logs."""
        return " ".join(
            f"{field.name}={getattr(self, field.name)}" for field in dataclasses.fields(self)
        )
