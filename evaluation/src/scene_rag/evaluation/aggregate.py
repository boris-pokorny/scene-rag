"""Roll-ups of per-case scores, overall and per tag.

A metric's `Aggregation` picks the roll-up: ``mean`` keeps the key, ``sum`` keeps the key,
``percentiles`` yields ``<key>.p50`` and ``<key>.p95`` (nearest rank). A key no case scored
is left out rather than reported as zero, so "not measured" never reads as "perfect".
"""

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from scene_rag.evaluation.models import Aggregates, Aggregation, CaseResult


def percentile(values: Sequence[float], p: float) -> float:
    """Return the nearest-rank ``p``-th percentile of non-empty ``values``."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _mean(key: str, values: Sequence[float]) -> dict[str, float]:
    return {key: sum(values) / len(values)}


def _sum(key: str, values: Sequence[float]) -> dict[str, float]:
    return {key: float(sum(values))}


def _percentiles(key: str, values: Sequence[float]) -> dict[str, float]:
    return {f"{key}.p50": percentile(values, 50), f"{key}.p95": percentile(values, 95)}


ROLL_UPS: Final[Mapping[Aggregation, Callable[[str, Sequence[float]], dict[str, float]]]] = {
    "mean": _mean,
    "sum": _sum,
    "percentiles": _percentiles,
}


def roll_up(results: Sequence[CaseResult], kinds: Mapping[str, Aggregation]) -> dict[str, float]:
    """Roll the scores of ``results`` up, one roll-up per metric key in ``kinds``."""
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, value in result.metrics.items():
            values[key].append(value)
    rolled: dict[str, float] = {}
    for key, kind in sorted(kinds.items()):
        if values.get(key):
            rolled.update(ROLL_UPS[kind](key, values[key]))
    return rolled


def aggregate(results: Sequence[CaseResult], kinds: Mapping[str, Aggregation]) -> Aggregates:
    """Roll the scores up over every case and over the cases of each tag."""
    tags = sorted({tag for result in results for tag in result.tags})
    return Aggregates(
        overall=roll_up(results, kinds),
        by_tag={tag: roll_up([r for r in results if tag in r.tags], kinds) for tag in tags},
    )
