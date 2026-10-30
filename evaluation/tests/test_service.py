"""The run use case over a fake query-api."""

import pytest

from scene_rag.evaluation.bootstrap import build_metrics
from scene_rag.evaluation.config import RunConfig
from scene_rag.evaluation.metrics import Leakage
from scene_rag.evaluation.models import Response, Served
from scene_rag.evaluation.protocols import RunError
from scene_rag.evaluation.service import RunService
from tests.support.fakes import (
    DATASET_REF,
    SERVED,
    T0,
    FakeQueryApi,
    MemoryRunWriter,
    Script,
    failing,
    make_answer,
    make_case,
    make_dataset,
    source,
    span,
    ticking_clock,
    ticking_timer,
)

ANSWERABLE = make_case("q-1", relevant=(span(10, 20),), tags=("visual",))
UNANSWERABLE = make_case("q-2", tags=("unanswerable",), entitlements=frozenset({"a", "b"}))


def found(question: str) -> Response:
    return Response((source(12, 14), source(50, 60)), SERVED)


def answered(question: str) -> Response:
    refused = question == UNANSWERABLE.question
    return Response((source(12, 14),), SERVED, make_answer(refused=refused))


def make_service(
    search: Script = found,
    answer: Script = answered,
    config: RunConfig | None = None,
) -> tuple[RunService, FakeQueryApi, MemoryRunWriter]:
    run_config = config or RunConfig()
    api = FakeQueryApi(search, answer)
    writer = MemoryRunWriter()
    service = RunService(
        query_api=api,
        metrics=build_metrics(run_config),
        writer=writer,
        config=run_config,
        clock=ticking_clock(),
        timer=ticking_timer(),
        new_run_id=lambda: "run-000000000001",
    )
    return service, api, writer


class TestGivenHealthyQueryApi:
    """Every call succeeds and serves the same index."""

    def test_when_run_then_stored_with_dataset_served_and_config(self) -> None:
        service, _, writer = make_service()

        run = service.run(make_dataset(ANSWERABLE, UNANSWERABLE))

        assert writer.runs == [run]
        assert run.run_id == "run-000000000001" and run.dataset == DATASET_REF
        assert run.served == SERVED and run.config == RunConfig() and run.errors == 0
        assert run.started_at == T0 and run.finished_at > run.started_at

    def test_when_run_then_each_case_asked_under_its_entitlements(self) -> None:
        service, api, _ = make_service()

        service.run(make_dataset(ANSWERABLE, UNANSWERABLE))

        assert api.calls == [
            ("search", ANSWERABLE.question, frozenset({"public"}), 10),
            ("answer", ANSWERABLE.question, frozenset({"public"}), None),
            ("search", UNANSWERABLE.question, frozenset({"a", "b"}), 10),
            ("answer", UNANSWERABLE.question, frozenset({"a", "b"}), None),
        ]

    def test_when_run_then_case_scores_keyed_by_use_case(self) -> None:
        service, _, _ = make_service()

        first, second = service.run(make_dataset(ANSWERABLE, UNANSWERABLE)).cases

        assert first.metrics["search/recall@5"] == 1.0
        assert first.metrics["search/mrr"] == 1.0
        assert first.metrics["answer/refusal_correctness"] == 1.0
        assert first.metrics["search/latency_ms"] == pytest.approx(50.0)
        assert "search/recall@5" not in second.metrics
        assert second.metrics["answer/refusal_correctness"] == 1.0
        assert second.tags == ("unanswerable",)

    def test_when_run_then_aggregates_overall_and_per_tag(self) -> None:
        service, _, _ = make_service()

        aggregates = service.run(make_dataset(ANSWERABLE, UNANSWERABLE)).aggregates

        assert aggregates.overall["search/recall@5"] == 1.0
        assert aggregates.overall["answer/prompt_tokens"] == 100.0
        assert aggregates.overall["search/latency_ms.p95"] == pytest.approx(50.0)
        assert aggregates.overall["search/leakage"] == 0.0
        assert set(aggregates.by_tag) == {"visual", "unanswerable"}
        assert "search/recall@5" not in aggregates.by_tag["unanswerable"]

    def test_when_only_search_configured_then_answer_never_called(self) -> None:
        service, api, _ = make_service(config=RunConfig(use_cases=("search",)))

        run = service.run(make_dataset(ANSWERABLE))

        assert [call[0] for call in api.calls] == ["search"]
        assert not any(key.startswith("answer/") for key in run.aggregates.overall)


class TestGivenFailingQueryApi:
    """A failed call is recorded on its case; the run carries on."""

    def test_when_answer_fails_then_error_recorded_and_search_scored(self) -> None:
        service, _, writer = make_service(answer=failing("HTTP 503 from /answer"))

        run = service.run(make_dataset(ANSWERABLE))

        assert run.errors == 1 and writer.runs == [run]
        assert run.cases[0].errors == ("answer: HTTP 503 from /answer",)
        assert "search/recall@5" in run.cases[0].metrics
        assert not any(key.startswith("answer/") for key in run.cases[0].metrics)

    def test_when_every_call_fails_then_nothing_served(self) -> None:
        service, _, _ = make_service(search=failing(), answer=failing())

        run = service.run(make_dataset(ANSWERABLE, UNANSWERABLE))

        assert run.served is None and run.errors == 4 and run.aggregates.overall == {}


class TestGivenReleaseChangingMidRun:
    """Scores across two releases mean nothing."""

    def test_when_served_changes_then_run_fails_and_nothing_stored(self) -> None:
        other = Served("22d0a9e2c4f1", "5e2d4c6b8a90", None, "chunks_5e2d4c6b8a90")

        def promoted(question: str) -> Response:
            return Response((), other if question == UNANSWERABLE.question else SERVED)

        service, _, writer = make_service(search=promoted)

        with pytest.raises(RunError, match="case q-2"):
            service.run(make_dataset(ANSWERABLE, UNANSWERABLE))
        assert writer.runs == []


class TestGivenMetricRegistry:
    """Metric keys are unique."""

    def test_when_keys_repeat_then_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            RunService(
                query_api=FakeQueryApi(found, answered),
                metrics=[Leakage("search"), Leakage("search")],
                writer=MemoryRunWriter(),
                config=RunConfig(),
                clock=ticking_clock(),
                timer=ticking_timer(),
                new_run_id=lambda: "run-000000000001",
            )
