"""Roll-ups of case scores, overall and per tag."""

import pytest

from scene_rag.evaluation.aggregate import aggregate, percentile, roll_up
from scene_rag.evaluation.models import Aggregation, CaseResult

KINDS: dict[str, Aggregation] = {
    "search/recall@5": "mean",
    "search/leakage": "sum",
    "search/latency_ms": "percentiles",
    "answer/refusal_correctness": "mean",
}
RESULTS = [
    CaseResult(
        "a", ("visual",), {"search/recall@5": 1.0, "search/leakage": 0.0, "search/latency_ms": 10.0}
    ),
    CaseResult(
        "b",
        ("visual", "temporal"),
        {"search/recall@5": 0.5, "search/leakage": 2.0, "search/latency_ms": 30.0},
    ),
    CaseResult("c", (), {"search/leakage": 0.0, "search/latency_ms": 20.0}),
]


class TestGivenPercentile:
    """Nearest rank."""

    @pytest.mark.parametrize(("p", "expected"), [(50, 2.0), (95, 4.0), (0, 1.0), (100, 4.0)])
    def test_when_computed_then_nearest_rank(self, p: float, expected: float) -> None:
        assert percentile([4.0, 1.0, 3.0, 2.0], p) == expected


class TestGivenResults:
    """Means, sums and percentiles; absent when no case scored the metric."""

    def test_when_rolled_up_then_each_kind_applies(self) -> None:
        assert roll_up(RESULTS, KINDS) == {
            "search/latency_ms.p50": 20.0,
            "search/latency_ms.p95": 30.0,
            "search/leakage": 2.0,
            "search/recall@5": 0.75,
        }

    def test_when_aggregated_then_per_tag_over_its_cases(self) -> None:
        aggregates = aggregate(RESULTS, KINDS)

        assert sorted(aggregates.by_tag) == ["temporal", "visual"]
        assert aggregates.by_tag["temporal"]["search/recall@5"] == 0.5
        assert aggregates.by_tag["visual"]["search/leakage"] == 2.0
        assert aggregates.overall == roll_up(RESULTS, KINDS)

    def test_when_no_results_then_empty(self) -> None:
        aggregates = aggregate([], KINDS)

        assert aggregates.overall == {} and aggregates.by_tag == {}
