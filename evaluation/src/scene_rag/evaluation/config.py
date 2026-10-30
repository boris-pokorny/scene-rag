"""Hyperparameters of the service: how a run measures, and what the gate demands.

The field defaults *are* the values. They belong to no release: they define how a release is
measured and judged, not what is served, so every eval run records its `RunConfig` and every
verdict its `GateConfig`. Nothing here comes from the environment; where query-api and the
Eval Store are lives in `settings.py`.
"""

import dataclasses
import math
from typing import Final, Literal

type UseCase = Literal["search", "answer"]
"""The query-api use cases a case is put to."""

USE_CASES: Final[tuple[UseCase, ...]] = ("search", "answer")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Every tunable of a run, and nothing else.

    Attributes:
        ks: Cut-offs of hit@k, recall@k and nDCG@k.
        top_k: Sources a search asks for; at least the largest cut-off. Answers are asked
            without a count, so the generator sees exactly what query-api serves by default.
        use_cases: Use cases every case is put to.
    """

    ks: tuple[int, ...] = (1, 5, 10)
    top_k: int = 10
    use_cases: tuple[UseCase, ...] = USE_CASES

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working run.
        """
        if not self.ks or min(self.ks) < 1:
            raise ValueError("ks must name at least one cut-off, each at least 1")
        if self.top_k < max(self.ks):
            raise ValueError("top_k must be at least the largest cut-off")
        if not self.use_cases or not set(self.use_cases) <= set(USE_CASES):
            raise ValueError(f"use_cases must be a non-empty subset of {list(USE_CASES)}")
        if len(set(self.use_cases)) != len(self.use_cases):
            raise ValueError("use_cases must not repeat")


@dataclasses.dataclass(frozen=True)
class GateConfig:
    """Every tunable of the promotion gate, and nothing else.

    Attributes:
        primary_metrics: Overall aggregates a candidate may not lose more than `tolerance`
            on against its baseline; it must report each of them.
        tolerance: Largest drop of a primary metric that still passes.
        zero_metrics: Overall aggregates that must be zero, baseline or not; absent counts
            as zero, since a leak no case could observe is no leak.
    """

    primary_metrics: tuple[str, ...] = ("search/recall@5",)
    tolerance: float = 0.02
    zero_metrics: tuple[str, ...] = ("search/leakage", "answer/leakage")

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working gate.
        """
        if not (math.isfinite(self.tolerance) and self.tolerance >= 0):
            raise ValueError("tolerance must be a finite number of at least 0")
