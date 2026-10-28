"""What a chunking pass reports: the outcome of each scene set and the pass's counters.

Orchestration vocabulary, deliberately outside the fingerprint: how a pass is tallied and
logged changes nothing about what a chunk contains.
"""

import dataclasses
from collections import Counter
from collections.abc import Iterable
from typing import Literal

type Outcome = Literal["chunked", "duplicate", "dropped", "deferred", "failed"]
"""What became of one scene set in a pass; the field names of `ChunkingReport`."""


@dataclasses.dataclass(frozen=True)
class ChunkingReport:
    """Counters of one chunking pass, per scene set.

    Attributes:
        chunked: Scene sets whose chunk set was written.
        duplicate: Scene sets whose chunk set another worker wrote first; nothing changed.
        dropped: Scene sets whose asset version was retired while being chunked.
        deferred: Scene sets not complete under the pins at fetch time; left pending.
        failed: Scene sets that raised; the pass continued past them.
    """

    chunked: int = 0
    duplicate: int = 0
    dropped: int = 0
    deferred: int = 0
    failed: int = 0

    @classmethod
    def counting(cls, outcomes: Iterable[Outcome]) -> ChunkingReport:
        """Tally the outcomes of a pass, in whatever order they arrive."""
        counts: Counter[str] = Counter(outcomes)
        return cls(**{field.name: counts[field.name] for field in dataclasses.fields(cls)})

    def __str__(self) -> str:
        """Render the counters as ``name=value`` pairs for logs."""
        return " ".join(
            f"{field.name}={getattr(self, field.name)}" for field in dataclasses.fields(self)
        )
