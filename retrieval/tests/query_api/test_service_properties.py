"""Properties of the use cases over any question, entitlements and generated answer."""

from hypothesis import given, settings
from hypothesis import strategies as st

from scene_rag.retrieval.query_api import models
from tests.support.query_fakes import (
    OTHER_ASSET,
    PUBLIC,
    TEAM,
    make_entry_view,
    make_query_harness,
)

ENTRIES = tuple(make_entry_view(i, entitlements=PUBLIC) for i in range(3)) + tuple(
    make_entry_view(i, asset_version=OTHER_ASSET, entitlements=TEAM) for i in range(3, 5)
)
names = st.sampled_from(["public", "team-platform", "other"])
entitlements = st.frozensets(names, max_size=3)
questions = st.text(min_size=1, max_size=40).filter(str.strip)
answers = st.text(max_size=60)


class TestGivenAnyRequest:
    """Whatever is asked, by whom, whatever the generator writes."""

    @settings(max_examples=60, deadline=None)
    @given(question=questions, caller=entitlements, text=answers, top_k=st.integers(1, 30))
    def test_when_answered_then_citations_name_sources_and_sources_are_visible(
        self, question: str, caller: frozenset[str], text: str, top_k: int
    ) -> None:
        harness = make_query_harness(*ENTRIES)
        harness.generator.text = text

        result = harness.service.answer(
            models.AnswerQuery(question, models.Caller(caller), top_k=top_k)
        )

        assert all(1 <= n <= len(result.sources) for n in result.citations)
        assert set(result.citations).isdisjoint(result.invalid_citations)
        assert all(s.entitlements & caller for s in result.sources)
        assert len(result.sources) <= min(top_k, harness.config.retriever.max_top_k)
        if result.refused:
            assert result.citations == () and harness.generator.prompts == []
            assert result.usage == models.NO_USAGE
        else:
            assert len(harness.generator.prompts) == 1
            assert result.answer == text
        if not caller:
            assert result.sources == () and harness.embedder.questions == []

    @settings(max_examples=40, deadline=None)
    @given(question=questions, caller=entitlements)
    def test_when_searched_then_scores_descend_and_match_answer_sources(
        self, question: str, caller: frozenset[str]
    ) -> None:
        harness = make_query_harness(*ENTRIES)

        found = harness.service.search(models.SearchQuery(question, models.Caller(caller)))
        answered = harness.service.answer(models.AnswerQuery(question, models.Caller(caller)))

        scores = [s.score for s in found.sources]
        assert scores == sorted(scores, reverse=True)
        assert found.sources == answered.sources
