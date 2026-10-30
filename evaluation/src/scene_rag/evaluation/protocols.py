"""Ports of the run use case: query-api, metrics, the judge and the Eval Store.

`RunService` depends on these Protocols only; `bootstrap` binds the implementations.
"""

from collections.abc import Sequence
from typing import Protocol

from scene_rag.evaluation.config import UseCase
from scene_rag.evaluation.models import (
    Aggregation,
    Answer,
    Case,
    EvalRun,
    Observation,
    Response,
    Source,
)


class QueryApiError(RuntimeError):
    """query-api could not be reached, refused the request, or answered undecodably."""


class RunError(RuntimeError):
    """The run cannot produce a meaningful result and stores nothing."""


class QueryApi(Protocol):
    """The published query API, as evaluation calls it."""

    def search(self, question: str, entitlements: frozenset[str], top_k: int) -> Response:
        """Return up to ``top_k`` sources the caller may see, best first.

        Raises:
            QueryApiError: When the call fails.
        """
        ...

    def answer(self, question: str, entitlements: frozenset[str]) -> Response:
        """Return a cited answer, or a refusal, under the served defaults.

        Raises:
            QueryApiError: When the call fails.
        """
        ...


class Metric(Protocol):
    """One score per case and use case.

    A new metric is a new class registered in `bootstrap`, never a branch in the runner.
    """

    @property
    def name(self) -> str:
        """Return the metric's name, unique within its use case, e.g. ``recall@5``."""
        ...

    @property
    def use_case(self) -> UseCase:
        """Return the use case whose observations the metric scores."""
        ...

    @property
    def aggregation(self) -> Aggregation:
        """Return how the per-case scores roll up."""
        ...

    def score(self, case: Case, observation: Observation) -> float | None:
        """Score one observation; ``None`` when the metric does not apply to the case."""
        ...


class Judge(Protocol):
    """Grades an answer with a model, for faithfulness and correctness.

    Not implemented yet: a judge must be a different model than the generator, and the
    service has no model backend. A judge will be wrapped in `Metric`s, so the runner does
    not change when one arrives.
    """

    def grade(self, case: Case, answer: Answer, sources: Sequence[Source]) -> dict[str, float]:
        """Return a score between 0 and 1 per judged quality."""
        ...


class EvalRunWriter(Protocol):
    """Stores eval runs."""

    def write(self, run: EvalRun) -> str:
        """Store ``run`` and return where it went, for the log."""
        ...
