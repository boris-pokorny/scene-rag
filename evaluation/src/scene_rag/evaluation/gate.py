"""The promotion gate: a pure decision over two eval runs.

A candidate passes when it had no failed call, every zero metric is zero, and no primary
metric dropped below its baseline by more than the tolerance. Runs over different datasets
are not comparable and are refused, rather than judged.
"""

from collections.abc import Mapping
from datetime import datetime

from scene_rag.evaluation.config import GateConfig
from scene_rag.evaluation.models import Check, EvalRun, MetricDelta, Verdict


class DatasetMismatchError(ValueError):
    """The two runs used different datasets, or different bytes under one version."""


def _scopes(run: EvalRun) -> dict[str, Mapping[str, float]]:
    scopes: dict[str, Mapping[str, float]] = {"overall": run.aggregates.overall}
    by_tag = run.aggregates.by_tag
    scopes.update({f"tag:{tag}": values for tag, values in sorted(by_tag.items())})
    return scopes


def deltas(candidate: EvalRun, baseline: EvalRun) -> list[MetricDelta]:
    """Return every aggregate of either run, overall first, then per tag."""
    candidate_scopes, baseline_scopes = _scopes(candidate), _scopes(baseline)
    result: list[MetricDelta] = []
    for scope in dict.fromkeys([*candidate_scopes, *baseline_scopes]):
        new, old = candidate_scopes.get(scope, {}), baseline_scopes.get(scope, {})
        for metric in sorted(new.keys() | old.keys()):
            result.append(MetricDelta(scope, metric, old.get(metric), new.get(metric)))
    return result


def _primary(metric: str, candidate: EvalRun, baseline: EvalRun | None, tolerance: float) -> Check:
    value = candidate.aggregates.overall.get(metric)
    reference = None if baseline is None else baseline.aggregates.overall.get(metric)
    passed = value is not None and (reference is None or value >= reference - tolerance)
    return Check(metric, value, reference, tolerance, passed)


def decide(
    candidate: EvalRun,
    baseline: EvalRun | None,
    config: GateConfig,
    decided_at: datetime,
) -> Verdict:
    """Apply the gate to ``candidate``, against ``baseline`` when there is one.

    Without a baseline (a first release) only the absolute rules can fail; a primary metric
    must still be reported.

    Raises:
        DatasetMismatchError: When the runs used different datasets.
    """
    if baseline is not None and baseline.dataset != candidate.dataset:
        raise DatasetMismatchError(
            f"{candidate.run_id} used {candidate.dataset}, {baseline.run_id} used "
            f"{baseline.dataset}; compare runs over the same dataset version"
        )
    checks = [Check("errors", float(candidate.errors), None, None, candidate.errors == 0)]
    for metric in config.zero_metrics:
        value = candidate.aggregates.overall.get(metric, 0.0)
        checks.append(Check(metric, value, None, None, value == 0))
    checks += [
        _primary(metric, candidate, baseline, config.tolerance) for metric in config.primary_metrics
    ]
    return Verdict(
        run_id=candidate.run_id,
        baseline_run_id=None if baseline is None else baseline.run_id,
        dataset=candidate.dataset,
        served=candidate.served,
        checks=tuple(checks),
        gate=config,
        decided_at=decided_at,
    )
