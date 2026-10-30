"""Domain models: dataset cases, what query-api answered, the eval run and the gate's verdict.

Every model is a frozen dataclass. The query-api documents are decoded into the few fields
evaluation reads (`query_api_client`); the rest of the published response is ignored, so a
backward-compatible addition there changes nothing here.
"""

import dataclasses
from collections.abc import Mapping
from datetime import datetime
from typing import Final, Literal

from scene_rag.evaluation.config import GateConfig, RunConfig, UseCase

SCHEMA_VERSION: Final = 1
"""Version of the eval run document's schema."""

VERDICT_SCHEMA_VERSION: Final = 1
"""Version of the published verdict document's schema."""

type Aggregation = Literal["mean", "sum", "percentiles"]
"""How a metric's per-case scores roll up: their mean, their sum, or their p50 and p95."""


@dataclasses.dataclass(frozen=True)
class Span:
    """A labelled time span of one asset version.

    Attributes:
        asset_id: Id of the asset in its source system; informational, never matched.
        asset_version: Content hash of the asset version; what a source is matched by.
        start_seconds: Start of the span.
        end_seconds: End of the span, after its start.
    """

    asset_id: str
    asset_version: str
    start_seconds: float
    end_seconds: float


@dataclasses.dataclass(frozen=True)
class Case:
    """One dataset case.

    Attributes:
        id: Unique within the dataset.
        question: The question, as a user would ask it.
        relevant: Spans a correct search finds; empty when the question is unanswerable
            and the correct answer is a refusal.
        forbidden: Spans the caller must never see; a returned source overlapping one is a
            leak.
        tags: Labels the aggregates are broken down by.
        caller_entitlements: Entitlements the case is asked under.
        reference_answer: The expected answer, for a judge; unused until one exists.
    """

    id: str
    question: str
    relevant: tuple[Span, ...]
    forbidden: tuple[Span, ...]
    tags: tuple[str, ...]
    caller_entitlements: frozenset[str]
    reference_answer: str | None = None

    @property
    def answerable(self) -> bool:
        """Return whether the dataset labels any relevant span."""
        return bool(self.relevant)


@dataclasses.dataclass(frozen=True)
class DatasetRef:
    """Which dataset a run used.

    Attributes:
        name: The dataset's directory name.
        version: Its semantic version; the major changes when label meaning changes.
        sha256: Hex digest of the file, so an edit without a version bump is detected.
    """

    name: str
    version: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class Dataset:
    """A loaded dataset.

    Attributes:
        ref: Which dataset it is.
        cases: Its cases, in file order.
    """

    ref: DatasetRef
    cases: tuple[Case, ...]


@dataclasses.dataclass(frozen=True)
class Source:
    """A source query-api returned, reduced to what relevance is judged by.

    Attributes:
        asset_version: Content hash of the asset version.
        start_seconds: Start of the span.
        end_seconds: End of the span.
    """

    asset_version: str
    start_seconds: float
    end_seconds: float


@dataclasses.dataclass(frozen=True)
class Served:
    """What query-api reported as having produced a response.

    Attributes:
        query_config: Id of the query configuration in force.
        embedding_fingerprint: Fingerprint of the index served.
        chunking_fingerprint: Chunking fingerprint the index accepted; ``None`` when any.
        collection: Collection of the Search Index served.
    """

    query_config: str
    embedding_fingerprint: str
    chunking_fingerprint: str | None
    collection: str


@dataclasses.dataclass(frozen=True)
class Answer:
    """The generation part of an answer response.

    Attributes:
        text: The answer, or the refusal text.
        refused: Whether query-api declined to answer.
        citations: Source numbers the answer cites validly.
        invalid_citations: Numbers the answer cites that name no source.
        prompt_tokens: Tokens of the prompt, when the model reports them.
        completion_tokens: Tokens of the answer, when the model reports them.
    """

    text: str
    refused: bool
    citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclasses.dataclass(frozen=True)
class Response:
    """A decoded query-api response.

    Attributes:
        sources: The sources, best first.
        served: What produced them.
        answer: The generation part; ``None`` for a search.
    """

    sources: tuple[Source, ...]
    served: Served
    answer: Answer | None = None


@dataclasses.dataclass(frozen=True)
class Observation:
    """One use case of one case, as the runner saw it.

    Attributes:
        use_case: Which use case was called.
        response: What query-api answered.
        latency_ms: Wall-clock time of the call, in milliseconds.
    """

    use_case: UseCase
    response: Response
    latency_ms: float


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """The scores of one case.

    Attributes:
        case_id: Id of the case.
        tags: The case's tags.
        metrics: Score per metric key (``<use case>/<metric>``); a metric that does not
            apply to the case is absent.
        errors: One message per use case query-api failed on.
    """

    case_id: str
    tags: tuple[str, ...]
    metrics: Mapping[str, float]
    errors: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Aggregates:
    """Roll-ups of the case scores.

    Attributes:
        overall: Value per aggregate key over every case.
        by_tag: The same per tag, over the cases carrying it.
    """

    overall: Mapping[str, float]
    by_tag: Mapping[str, Mapping[str, float]]


@dataclasses.dataclass(frozen=True)
class EvalRun:
    """One evaluation of what query-api serves, against one dataset version.

    Attributes:
        run_id: ``run-`` and twelve hex digits.
        started_at: When the first case was put.
        finished_at: When the last case was scored.
        dataset: Which dataset was used.
        served: What query-api served throughout; ``None`` when no call succeeded.
        config: The run's own hyperparameters; they belong to no release.
        cases: Per-case scores, in dataset order.
        aggregates: Their roll-ups.
        errors: Failed query-api calls.
        schema_version: Version of the stored document's schema.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    dataset: DatasetRef
    served: Served | None
    config: RunConfig
    cases: tuple[CaseResult, ...]
    aggregates: Aggregates
    errors: int
    schema_version: int = SCHEMA_VERSION


@dataclasses.dataclass(frozen=True)
class MetricDelta:
    """One aggregate in two runs.

    Attributes:
        scope: ``overall`` or ``tag:<tag>``.
        metric: The aggregate key.
        baseline: Its value in the baseline; ``None`` when absent there.
        candidate: Its value in the candidate; ``None`` when absent there.
    """

    scope: str
    metric: str
    baseline: float | None
    candidate: float | None

    @property
    def delta(self) -> float | None:
        """Return candidate minus baseline; ``None`` unless both are present."""
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline


@dataclasses.dataclass(frozen=True)
class Check:
    """One rule of the gate, applied.

    Attributes:
        name: What was checked: ``errors`` or an aggregate key.
        candidate: The candidate's value; ``None`` when it did not report it.
        baseline: The baseline's value; ``None`` without a baseline or for absolute rules.
        tolerance: The allowed drop; ``None`` for absolute rules.
        passed: Whether the rule holds.
    """

    name: str
    candidate: float | None
    baseline: float | None
    tolerance: float | None
    passed: bool


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The promotion gate's decision on a candidate run.

    Attributes:
        run_id: The candidate run.
        baseline_run_id: The run it was held against; ``None`` for a first release.
        dataset: The dataset both runs used.
        served: What the candidate run measured.
        checks: Every rule applied.
        gate: The rules' parameters.
        decided_at: When the verdict was reached.
        schema_version: Version of the published document's schema.
    """

    run_id: str
    baseline_run_id: str | None
    dataset: DatasetRef
    served: Served | None
    checks: tuple[Check, ...]
    gate: GateConfig
    decided_at: datetime
    schema_version: int = VERDICT_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        """Return whether every check passed."""
        return all(check.passed for check in self.checks)
