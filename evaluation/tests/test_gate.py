"""The promotion gate, a pure decision."""

import dataclasses

import pytest

from scene_rag.evaluation.config import GateConfig
from scene_rag.evaluation.gate import DatasetMismatchError, decide, deltas
from scene_rag.evaluation.models import EvalRun, MetricDelta
from tests.support.fakes import DATASET_REF, T0, make_run

OVERALL = {"search/recall@5": 0.80, "search/leakage": 0.0, "search/mrr": 0.5}
BY_TAG = {"visual": {"search/recall@5": 0.7}}
BASELINE = make_run("run-000000000001", OVERALL, BY_TAG)


def candidate(
    overall: dict[str, float] | None = None,
    by_tag: dict[str, dict[str, float]] | None = None,
    errors: int = 0,
    sha256: str = DATASET_REF.sha256,
    version: str = DATASET_REF.version,
) -> EvalRun:
    dataset = dataclasses.replace(DATASET_REF, sha256=sha256, version=version)
    return make_run(
        "run-000000000002",
        OVERALL if overall is None else overall,
        BY_TAG if by_tag is None else by_tag,
        errors,
        dataset,
    )


def with_overall(values: dict[str, float]) -> EvalRun:
    return candidate(overall={**OVERALL, **values})


class TestGivenCandidateAndBaseline:
    """Relative and absolute rules."""

    def test_when_equal_then_passed_with_every_check(self) -> None:
        verdict = decide(candidate(), BASELINE, GateConfig(), T0)

        assert verdict.passed and verdict.baseline_run_id == BASELINE.run_id
        assert [c.name for c in verdict.checks] == [
            "errors",
            "search/leakage",
            "answer/leakage",
            "search/recall@5",
        ]
        assert verdict.checks[-1].baseline == 0.80 and verdict.checks[-1].tolerance == 0.02

    @pytest.mark.parametrize(("recall", "passed"), [(0.79, True), (0.78, True), (0.7799, False)])
    def test_when_primary_drops_then_tolerance_decides(self, recall: float, passed: bool) -> None:
        verdict = decide(with_overall({"search/recall@5": recall}), BASELINE, GateConfig(), T0)

        assert verdict.passed is passed

    def test_when_leaking_then_failed(self) -> None:
        verdict = decide(with_overall({"search/leakage": 1.0}), BASELINE, GateConfig(), T0)

        assert not verdict.passed
        assert [c.name for c in verdict.checks if not c.passed] == ["search/leakage"]

    def test_when_calls_failed_then_failed(self) -> None:
        verdict = decide(candidate(errors=2), BASELINE, GateConfig(), T0)

        assert [c.name for c in verdict.checks if not c.passed] == ["errors"]

    def test_when_primary_missing_then_failed(self) -> None:
        verdict = decide(candidate(overall={"search/leakage": 0.0}), BASELINE, GateConfig(), T0)

        assert verdict.checks[-1].candidate is None and not verdict.passed

    def test_when_baseline_lacks_primary_then_only_presence_needed(self) -> None:
        baseline = make_run("run-000000000001")

        assert decide(candidate(), baseline, GateConfig(), T0).passed

    def test_when_datasets_differ_then_refused(self) -> None:
        bumped = candidate(version="2.0.0")

        with pytest.raises(DatasetMismatchError, match="same dataset version"):
            decide(bumped, BASELINE, GateConfig(), T0)

    def test_when_bytes_differ_under_one_version_then_refused(self) -> None:
        edited = candidate(sha256="e" * 64)

        with pytest.raises(DatasetMismatchError):
            decide(edited, BASELINE, GateConfig(), T0)


class TestGivenNoBaseline:
    """A first release: only the absolute rules."""

    def test_when_no_baseline_then_absolute_rules_only(self) -> None:
        verdict = decide(with_overall({"search/recall@5": 0.1}), None, GateConfig(), T0)

        assert verdict.passed and verdict.baseline_run_id is None
        assert verdict.checks[-1].baseline is None


class TestGivenGateConfig:
    """Tolerance validates."""

    @pytest.mark.parametrize("tolerance", [-0.1, float("nan")])
    def test_when_invalid_then_rejected(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            GateConfig(tolerance=tolerance)


class TestGivenDeltas:
    """Every aggregate of either run, overall first."""

    def test_when_diffed_then_union_with_differences(self) -> None:
        new = candidate(
            overall={"search/recall@5": 0.9, "search/hit@1": 1.0},
            by_tag={"visual": {"search/recall@5": 0.75}, "audio": {"search/mrr": 1.0}},
        )

        rows = deltas(new, BASELINE)

        assert rows[0] == MetricDelta("overall", "search/hit@1", None, 1.0)
        assert rows[0].delta is None
        recall = next(r for r in rows if r.scope == "overall" and r.metric == "search/recall@5")
        assert recall.delta == pytest.approx(0.1)
        assert [r.scope for r in rows].index("tag:audio") < [r.scope for r in rows].index(
            "tag:visual"
        )
