"""The shipped metrics and span-overlap relevance, with hand-computed values."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
from scene_rag.evaluation.models import Source
from scene_rag.evaluation.protocols import Metric
from scene_rag.evaluation.relevance import covered, gains, overlaps
from tests.support.fakes import (
    VERSION_B,
    answer_observation,
    make_answer,
    make_case,
    search_observation,
    source,
    span,
)

ANSWERABLE = make_case(relevant=(span(10, 20), span(30, 40)))
UNANSWERABLE = make_case()


class TestGivenOverlap:
    """A source is relevant when it shares asset version and time with a span."""

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (5, 11, True),
            (19, 25, True),
            (12, 18, True),
            (0, 30, True),
            (0, 10, False),
            (20, 30, False),
        ],
    )
    def test_when_times_compared_then_touching_is_not_overlapping(
        self, start: float, end: float, expected: bool
    ) -> None:
        assert overlaps(source(start, end), span(10, 20)) is expected

    def test_when_other_version_then_no_overlap(self) -> None:
        assert not overlaps(source(10, 20, VERSION_B), span(10, 20))

    def test_when_spans_covered_then_indices_named(self) -> None:
        assert covered([source(0, 50)], ANSWERABLE.relevant) == {0, 1}

    def test_when_two_sources_on_one_span_then_credited_once(self) -> None:
        assert gains([source(10, 12), source(12, 14), source(35, 36)], ANSWERABLE.relevant) == [
            1,
            0,
            1,
        ]


class TestGivenRetrievalMetrics:
    """hit@k, recall@k, MRR and nDCG@k over a ranked list."""

    def test_when_relevant_second_then_scores_follow(self) -> None:
        observation = search_observation(source(50, 60), source(10, 20), source(60, 70))

        assert HitAtK(1).score(ANSWERABLE, observation) == 0.0
        assert HitAtK(5).score(ANSWERABLE, observation) == 1.0
        assert RecallAtK(5).score(ANSWERABLE, observation) == 0.5
        assert ReciprocalRank().score(ANSWERABLE, observation) == 0.5
        expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        assert NdcgAtK(5).score(ANSWERABLE, observation) == pytest.approx(expected)

    def test_when_nothing_relevant_then_zero(self) -> None:
        observation = search_observation(source(50, 60))

        assert ReciprocalRank().score(ANSWERABLE, observation) == 0.0
        assert NdcgAtK(5).score(ANSWERABLE, observation) == 0.0

    def test_when_perfect_then_one(self) -> None:
        observation = search_observation(source(10, 20), source(30, 40))

        assert RecallAtK(5).score(ANSWERABLE, observation) == 1.0
        assert NdcgAtK(5).score(ANSWERABLE, observation) == pytest.approx(1.0)

    def test_when_unanswerable_then_retrieval_metrics_do_not_apply(self) -> None:
        observation = search_observation(source(10, 20))
        metrics: list[Metric] = [HitAtK(5), RecallAtK(5), ReciprocalRank(), NdcgAtK(5)]

        assert [metric.score(UNANSWERABLE, observation) for metric in metrics] == [None] * 4

    def test_when_named_then_cut_off_in_name(self) -> None:
        assert [HitAtK(5).name, RecallAtK(1).name, NdcgAtK(10).name, ReciprocalRank().name] == [
            "hit@5",
            "recall@1",
            "ndcg@10",
            "mrr",
        ]

    @given(
        st.lists(
            st.tuples(st.integers(0, 50), st.integers(1, 20)).map(
                lambda t: source(t[0], t[0] + t[1])
            ),
            max_size=12,
        ),
        st.integers(1, 10),
    )
    def test_when_any_ranking_then_scores_between_zero_and_one(
        self, sources: list[Source], k: int
    ) -> None:
        observation = search_observation(*sources)
        for metric in (HitAtK(k), RecallAtK(k), NdcgAtK(k), ReciprocalRank()):
            value = metric.score(ANSWERABLE, observation)
            assert value is not None and 0.0 <= value <= 1.0 + 1e-12


class TestGivenLeakage:
    """A source overlapping a forbidden span is a leak, answerable or not."""

    def test_when_sources_overlap_forbidden_then_counted(self) -> None:
        case = make_case(forbidden=(span(0, 10, VERSION_B),))
        observation = search_observation(
            source(5, 6, VERSION_B), source(5, 6), source(9, 12, VERSION_B)
        )

        assert Leakage("search").score(case, observation) == 2.0
        assert Leakage("search").aggregation == "sum"

    def test_when_nothing_forbidden_then_zero(self) -> None:
        assert Leakage("answer").score(ANSWERABLE, search_observation(source(0, 1))) == 0.0


class TestGivenGenerationMetrics:
    """Citation validity, refusal correctness and tokens, deterministically."""

    def test_when_citations_invalid_then_zero(self) -> None:
        assert CitationValidity().score(ANSWERABLE, answer_observation(make_answer())) == 1.0
        observation = answer_observation(make_answer(invalid=(4,)))
        assert CitationValidity().score(ANSWERABLE, observation) == 0.0

    def test_when_refused_then_citation_validity_does_not_apply(self) -> None:
        observation = answer_observation(make_answer(refused=True))

        assert CitationValidity().score(UNANSWERABLE, observation) is None

    @pytest.mark.parametrize(
        ("answerable", "refused", "expected"),
        [(True, False, 1.0), (True, True, 0.0), (False, True, 1.0), (False, False, 0.0)],
    )
    def test_when_refusal_judged_then_correct_only_for_unanswerable(
        self, answerable: bool, refused: bool, expected: float
    ) -> None:
        case = ANSWERABLE if answerable else UNANSWERABLE
        observation = answer_observation(make_answer(refused=refused))

        assert RefusalCorrectness().score(case, observation) == expected

    def test_when_tokens_reported_then_returned(self) -> None:
        observation = answer_observation(make_answer(prompt_tokens=412, completion_tokens=23))

        assert Tokens("prompt").score(ANSWERABLE, observation) == 412.0
        assert Tokens("completion").score(ANSWERABLE, observation) == 23.0
        assert Tokens("prompt").name == "prompt_tokens"

    def test_when_tokens_unreported_then_do_not_apply(self) -> None:
        observation = answer_observation(make_answer(refused=True))

        assert Tokens("prompt").score(UNANSWERABLE, observation) is None

    def test_when_search_observed_then_answer_metrics_do_not_apply(self) -> None:
        observation = search_observation(source(10, 20))
        metrics: list[Metric] = [CitationValidity(), RefusalCorrectness(), Tokens("completion")]

        assert [metric.score(ANSWERABLE, observation) for metric in metrics] == [None] * 3

    def test_when_latency_scored_then_observed_value(self) -> None:
        observation = search_observation(latency_ms=42.0)

        assert Latency("search").score(ANSWERABLE, observation) == 42.0
        assert Latency("search").aggregation == "percentiles"
