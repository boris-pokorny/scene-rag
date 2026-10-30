"""The run use case: put every case to query-api, score it, roll up, store.

Orchestration only: what a score means lives in the metrics, how query-api is reached in its
client, where a run goes in the writer. A failed call is recorded on its case and the run
carries on, since one flaky answer should not discard an hour of judging; the gate counts the
errors. A change of what is served during the run fails it instead, because its scores would
mix two releases.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from scene_rag.evaluation.aggregate import aggregate
from scene_rag.evaluation.config import USE_CASES, RunConfig, UseCase
from scene_rag.evaluation.models import (
    Case,
    CaseResult,
    Dataset,
    EvalRun,
    Observation,
    Response,
    Served,
)
from scene_rag.evaluation.protocols import (
    EvalRunWriter,
    Metric,
    QueryApi,
    QueryApiError,
    RunError,
)

logger = logging.getLogger(__name__)

type Clock = Callable[[], datetime]
type Timer = Callable[[], float]
"""Monotonic seconds, for latencies."""
type RunIdFactory = Callable[[], str]


def metric_key(metric: Metric) -> str:
    """Return ``<use case>/<metric name>``, the key a score is stored under."""
    return f"{metric.use_case}/{metric.name}"


class RunService:
    """Evaluates query-api against a dataset."""

    def __init__(
        self,
        query_api: QueryApi,
        metrics: Sequence[Metric],
        writer: EvalRunWriter,
        config: RunConfig,
        clock: Clock,
        timer: Timer,
        new_run_id: RunIdFactory,
    ) -> None:
        """Bind the collaborators.

        Raises:
            ValueError: When two metrics share a key.
        """
        keys = [metric_key(metric) for metric in metrics]
        if len(set(keys)) != len(keys):
            raise ValueError(f"metric keys must be unique: {sorted(keys)}")
        self._metrics: Mapping[UseCase, list[Metric]] = {
            use_case: [metric for metric in metrics if metric.use_case == use_case]
            for use_case in USE_CASES
        }
        self._kinds = {metric_key(metric): metric.aggregation for metric in metrics}
        self._writer = writer
        self._config = config
        self._clock = clock
        self._timer = timer
        self._new_run_id = new_run_id
        self._calls: Mapping[UseCase, Callable[[Case], Response]] = {
            "search": lambda case: query_api.search(
                case.question, case.caller_entitlements, config.top_k
            ),
            "answer": lambda case: query_api.answer(case.question, case.caller_entitlements),
        }

    def run(self, dataset: Dataset) -> EvalRun:
        """Evaluate every case, store the run and return it.

        Raises:
            RunError: When what query-api serves changes during the run; nothing is stored.
        """
        run_id = self._new_run_id()
        started_at = self._clock()
        logger.info("run %s: %d cases of %s", run_id, len(dataset.cases), dataset.ref)
        served: Served | None = None
        results: list[CaseResult] = []
        for case in dataset.cases:
            result, observations = self._evaluate(case)
            for observation in observations:
                if served is None:
                    served = observation.response.served
                elif observation.response.served != served:
                    raise RunError(
                        f"query-api changed what it serves during the run (case {case.id}): "
                        f"{served} became {observation.response.served}"
                    )
            results.append(result)
        run = EvalRun(
            run_id=run_id,
            started_at=started_at,
            finished_at=self._clock(),
            dataset=dataset.ref,
            served=served,
            config=self._config,
            cases=tuple(results),
            aggregates=aggregate(results, self._kinds),
            errors=sum(len(result.errors) for result in results),
        )
        location = self._writer.write(run)
        logger.info("run %s stored at %s (%d errors)", run_id, location, run.errors)
        return run

    def _evaluate(self, case: Case) -> tuple[CaseResult, list[Observation]]:
        scores: dict[str, float] = {}
        errors: list[str] = []
        observations: list[Observation] = []
        for use_case in self._config.use_cases:
            start = self._timer()
            try:
                response = self._calls[use_case](case)
            except QueryApiError as exc:
                logger.warning("case %s, %s failed: %s", case.id, use_case, exc)
                errors.append(f"{use_case}: {exc}")
                continue
            latency_ms = round((self._timer() - start) * 1000, 3)
            observation = Observation(use_case, response, latency_ms)
            observations.append(observation)
            for metric in self._metrics[use_case]:
                value = metric.score(case, observation)
                if value is not None:
                    scores[metric_key(metric)] = value
        return CaseResult(case.id, case.tags, scores, tuple(errors)), observations
