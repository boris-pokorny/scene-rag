"""Composition root of the service's two commands.

`run` binds settings from the environment and hyperparameters from `RunConfig`, checks that
query-api answers, registers the metrics, wires `RunService`, and runs one dataset.
`compare` reads two runs back from the Eval Store, prints their differences, applies the gate
from `GateConfig` and writes the verdict. Nothing else reads the environment, and nothing
else names a backend or a metric.
"""

import logging
import os
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from scene_rag.evaluation import settings as settings_module
from scene_rag.evaluation.config import GateConfig, RunConfig
from scene_rag.evaluation.dataset import DatasetError, load_dataset
from scene_rag.evaluation.eval_store import EvalStoreError, JsonEvalStore
from scene_rag.evaluation.gate import DatasetMismatchError, decide, deltas
from scene_rag.evaluation.http_transport import JsonTransport, UrllibTransport
from scene_rag.evaluation.metrics import (
    CitationValidity,
    HitAtK,
    Latency,
    Leakage,
    NdcgAtK,
    RecallAtK,
    ReciprocalRank,
    RefusalCorrectness,
    Tokens,
)
from scene_rag.evaluation.models import EvalRun, MetricDelta, Verdict
from scene_rag.evaluation.protocols import EvalRunWriter, Metric, QueryApiError, RunError
from scene_rag.evaluation.query_api_client import QueryApiClient
from scene_rag.evaluation.service import RunService
from scene_rag.evaluation.settings import RunSettings

logger = logging.getLogger(__name__)

type Clock = Callable[[], datetime]


def _require(environ: Mapping[str, str], name: str, hint: str) -> str:
    value = environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. {hint}")
    return value


def load_eval_store_root(environ: Mapping[str, str] | None = None) -> Path:
    """Bind the Eval Store's directory, the one setting both commands need.

    Raises:
        RuntimeError: When it is not set.
    """
    env = os.environ if environ is None else environ
    return Path(
        _require(
            env,
            settings_module.EVAL_STORE_ROOT_ENV,
            "Name the Eval Store directory, e.g. data/eval-store.",
        )
    )


def load_settings(environ: Mapping[str, str] | None = None) -> RunSettings:
    """Bind `RunSettings` from the environment.

    Args:
        environ: Variables to read; the process environment by default.

    Returns:
        Validated settings.

    Raises:
        RuntimeError: When a variable is missing or the settings do not validate.
    """
    env = os.environ if environ is None else environ
    timeout_value = env.get(settings_module.QUERY_API_TIMEOUT_SECONDS_ENV)
    try:
        timeout = (
            float(timeout_value)
            if timeout_value
            else settings_module.DEFAULT_QUERY_API_TIMEOUT_SECONDS
        )
    except ValueError:
        raise RuntimeError(
            f"{settings_module.QUERY_API_TIMEOUT_SECONDS_ENV} must be a number, "
            f"not {timeout_value!r}"
        ) from None
    query_api_url = _require(
        env,
        settings_module.QUERY_API_URL_ENV,
        "Start the query-api of retrieval and point this at it, e.g. http://127.0.0.1:8081.",
    )
    root = load_eval_store_root(env)
    try:
        return RunSettings(
            query_api_url=query_api_url,
            eval_store_root=root,
            query_api_timeout_seconds=timeout,
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid settings: {exc}") from exc


def build_metrics(config: RunConfig) -> list[Metric]:
    """Register the shipped metrics for the configured cut-offs."""
    metrics: list[Metric] = []
    for k in config.ks:
        metrics += [HitAtK(k), RecallAtK(k), NdcgAtK(k)]
    metrics += [ReciprocalRank(), CitationValidity(), RefusalCorrectness()]
    metrics += [Tokens("prompt"), Tokens("completion")]
    for use_case in ("search", "answer"):
        metrics += [Leakage(use_case), Latency(use_case)]
    return metrics


def _new_run_id() -> str:
    return f"run-{secrets.token_hex(6)}"


def build_service(
    settings: RunSettings,
    config: RunConfig | None = None,
    transport: JsonTransport | None = None,
    writer: EvalRunWriter | None = None,
) -> RunService:
    """Wire `RunService`, after checking that query-api answers.

    Raises:
        RuntimeError: When query-api does not answer.
    """
    run_config = RunConfig() if config is None else config
    client = QueryApiClient(
        settings.query_api_url,
        UrllibTransport() if transport is None else transport,
        settings.query_api_timeout_seconds,
    )
    try:
        client.health()
    except QueryApiError as exc:
        raise RuntimeError(
            f"query-api at {settings.redacted_query_api_url} does not answer: {exc}"
        ) from exc
    return RunService(
        query_api=client,
        metrics=build_metrics(run_config),
        writer=JsonEvalStore(settings.eval_store_root) if writer is None else writer,
        config=run_config,
        clock=lambda: datetime.now(UTC),
        timer=time.perf_counter,
        new_run_id=_new_run_id,
    )


def format_report(run: EvalRun) -> list[str]:
    """Return the lines summarising ``run`` for the terminal."""
    lines = [f"{run.run_id}  dataset {run.dataset.name}@{run.dataset.version}"]
    if run.served is not None:
        lines.append(
            f"served  query_config {run.served.query_config}  "
            f"index {run.served.embedding_fingerprint}"
        )
    lines += [_line(key, value) for key, value in run.aggregates.overall.items()]
    lines.append(f"errors  {run.errors}")
    return lines


def _line(key: str, value: float) -> str:
    shown = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"  {key:<32} {shown}"


def run(dataset_path: Path, environ: Mapping[str, str] | None = None) -> EvalRun:
    """Evaluate query-api against one dataset file and print the summary.

    Raises:
        RuntimeError: When the settings, the dataset or the run are unusable.
    """
    settings = load_settings(environ)
    try:
        dataset = load_dataset(dataset_path)
    except DatasetError as exc:
        raise RuntimeError(f"invalid dataset: {exc}") from exc
    service = build_service(settings)
    try:
        result = service.run(dataset)
    except RunError as exc:
        raise RuntimeError(str(exc)) from exc
    _print(format_report(result))
    return result


def _print(lines: Sequence[str]) -> None:
    print("\n".join(lines))


def _value(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


def format_deltas(rows: Sequence[MetricDelta]) -> list[str]:
    """Return a table of the aggregates of two runs, with their differences."""
    lines = [f"{'scope':<16} {'metric':<32} {'baseline':>10} {'candidate':>10} {'delta':>10}"]
    for row in rows:
        delta = row.delta
        shown = "-" if delta is None else f"{delta:+.4f}"
        lines.append(
            f"{row.scope:<16} {row.metric:<32} {_value(row.baseline):>10} "
            f"{_value(row.candidate):>10} {shown:>10}"
        )
    return lines


def format_verdict(verdict: Verdict) -> list[str]:
    """Return the checks and the verdict, one line each."""
    lines = [
        f"{'ok  ' if check.passed else 'FAIL'} {check.name}: {_value(check.candidate)}"
        + ("" if check.baseline is None else f" (baseline {_value(check.baseline)})")
        for check in verdict.checks
    ]
    lines.append(f"verdict: {'passed' if verdict.passed else 'failed'}")
    return lines


def compare(
    candidate_id: str,
    baseline_id: str | None,
    environ: Mapping[str, str] | None = None,
    config: GateConfig | None = None,
    clock: Clock | None = None,
) -> Verdict:
    """Decide the gate for ``candidate_id`` against ``baseline_id``, print and store it.

    Raises:
        RuntimeError: When a run cannot be read or the runs are not comparable.
    """
    store = JsonEvalStore(load_eval_store_root(environ))
    try:
        candidate = store.read(candidate_id)
        baseline = None if baseline_id is None else store.read(baseline_id)
        verdict = decide(
            candidate,
            baseline,
            GateConfig() if config is None else config,
            (clock or (lambda: datetime.now(UTC)))(),
        )
    except (EvalStoreError, DatasetMismatchError) as exc:
        raise RuntimeError(str(exc)) from exc
    lines = [] if baseline is None else [*format_deltas(deltas(candidate, baseline)), ""]
    _print([*lines, *format_verdict(verdict)])
    location = store.write_verdict(verdict)
    logger.info("verdict for %s stored at %s", verdict.run_id, location)
    return verdict
