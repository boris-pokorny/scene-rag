"""Builders and fakes shared by the run and compare tests."""

import dataclasses
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

from scene_rag.evaluation.config import RunConfig
from scene_rag.evaluation.models import (
    Aggregates,
    Answer,
    Case,
    Dataset,
    DatasetRef,
    EvalRun,
    Observation,
    Response,
    Served,
    Source,
    Span,
)
from scene_rag.evaluation.protocols import QueryApiError

VERSION_A = "sha256:" + "a" * 64
VERSION_B = "sha256:" + "b" * 64
SERVED = Served("11d0a9e2c4f1", "5e2d4c6b8a90", None, "chunks_5e2d4c6b8a90")
DATASET_REF = DatasetRef("example", "1.0.0", "f" * 64)
T0 = datetime(2026, 10, 30, 10, 0, tzinfo=UTC)


def span(start: float, end: float, version: str = VERSION_A) -> Span:
    return Span("asset-1", version, start, end)


def source(start: float, end: float, version: str = VERSION_A) -> Source:
    return Source(version, start, end)


def make_case(
    case_id: str = "q-1",
    relevant: tuple[Span, ...] = (),
    forbidden: tuple[Span, ...] = (),
    tags: tuple[str, ...] = (),
    entitlements: frozenset[str] = frozenset({"public"}),
) -> Case:
    return Case(
        id=case_id,
        question=f"question {case_id}?",
        relevant=relevant,
        forbidden=forbidden,
        tags=tags,
        caller_entitlements=entitlements,
    )


def make_answer(
    refused: bool = False,
    invalid: tuple[int, ...] = (),
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 10,
) -> Answer:
    return Answer(
        text="refused" if refused else "An answer [1].",
        refused=refused,
        citations=() if refused else (1,),
        invalid_citations=invalid,
        prompt_tokens=None if refused else prompt_tokens,
        completion_tokens=None if refused else completion_tokens,
    )


def search_observation(*sources: Source, latency_ms: float = 10.0) -> Observation:
    return Observation("search", Response(tuple(sources), SERVED), latency_ms)


def answer_observation(answer: Answer, *sources: Source, latency_ms: float = 100.0) -> Observation:
    return Observation("answer", Response(tuple(sources), SERVED, answer), latency_ms)


def make_dataset(*cases: Case) -> Dataset:
    return Dataset(DATASET_REF, tuple(cases))


type Script = Callable[[str], Response]
"""Maps a question to a response, or raises QueryApiError."""


@dataclasses.dataclass
class FakeQueryApi:
    """Answers from per-use-case scripts and records every call."""

    search_script: Script
    answer_script: Script
    calls: list[tuple[str, str, frozenset[str], int | None]] = dataclasses.field(
        default_factory=list
    )

    def search(self, question: str, entitlements: frozenset[str], top_k: int) -> Response:
        self.calls.append(("search", question, entitlements, top_k))
        return self.search_script(question)

    def answer(self, question: str, entitlements: frozenset[str]) -> Response:
        self.calls.append(("answer", question, entitlements, None))
        return self.answer_script(question)


def failing(message: str = "HTTP 503") -> Script:
    def script(question: str) -> Response:
        raise QueryApiError(message)

    return script


@dataclasses.dataclass
class MemoryRunWriter:
    """Keeps written runs."""

    runs: list[EvalRun] = dataclasses.field(default_factory=list)

    def write(self, run: EvalRun) -> str:
        self.runs.append(run)
        return f"memory://{run.run_id}"


def ticking_clock(
    start: datetime = T0, step: timedelta = timedelta(seconds=1)
) -> Callable[[], datetime]:
    ticks: Iterator[int] = iter(range(10**6))
    return lambda: start + step * next(ticks)


def ticking_timer(step: float = 0.05) -> Callable[[], float]:
    """Each call advances 50 ms, so every observed latency is 50 ms."""
    ticks: Iterator[int] = iter(range(10**6))
    return lambda: step * next(ticks)


DEFAULT_CONFIG = RunConfig()


def make_run(
    run_id: str = "run-000000000001",
    overall: dict[str, float] | None = None,
    by_tag: dict[str, dict[str, float]] | None = None,
    errors: int = 0,
    dataset: DatasetRef = DATASET_REF,
) -> EvalRun:
    """A stored run as `compare` reads it back; cases do not matter to the gate."""
    return EvalRun(
        run_id=run_id,
        started_at=T0,
        finished_at=T0,
        dataset=dataset,
        served=SERVED,
        config=DEFAULT_CONFIG,
        cases=(),
        aggregates=Aggregates(overall or {}, by_tag or {}),
        errors=errors,
    )
