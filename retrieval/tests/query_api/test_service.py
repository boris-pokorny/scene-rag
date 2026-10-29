"""The use cases over the harness: validation, visibility, refusal, citations, resources."""

import pytest

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.config import QueryConfig, RetrieverConfig
from scene_rag.retrieval.query_api.protocols import (
    InvalidQueryError,
    ModelError,
    NotFoundError,
    SceneApiError,
)
from tests.support.query_fakes import (
    ASSET,
    OTHER_ASSET,
    PUBLIC,
    TEAM,
    FakeAnswerGenerator,
    FakeRetiredFilter,
    QueryHarness,
    make_entry_view,
)

PUBLIC_CALLER = models.Caller(PUBLIC)
ENTRIES = (
    make_entry_view(0),
    make_entry_view(1),
    make_entry_view(2, asset_version=OTHER_ASSET, entitlements=TEAM, text="team only"),
)


def search(question: str = "chunk 1 text", **kwargs: object) -> models.SearchQuery:
    return models.SearchQuery(question=question, caller=PUBLIC_CALLER, **kwargs)  # type: ignore[arg-type]


def answer(question: str = "chunk 1 text", **kwargs: object) -> models.AnswerQuery:
    return models.AnswerQuery(question=question, caller=PUBLIC_CALLER, **kwargs)  # type: ignore[arg-type]


class TestGivenIndexedEntries:
    """Two public entries of one asset, one team entry of another."""

    @pytest.fixture
    def query_harness(self, query_harness: QueryHarness) -> QueryHarness:
        query_harness.seed(*ENTRIES)
        return query_harness

    def test_when_searched_then_visible_sources_best_first_with_served(
        self, query_harness: QueryHarness
    ) -> None:
        result = query_harness.service.search(search())

        assert [s.chunk_version for s in result.sources][0] == ENTRIES[1].chunk_version
        assert {s.chunk_version for s in result.sources} == {e.chunk_version for e in ENTRIES[:2]}
        assert result.served == query_harness.served
        assert query_harness.search_index.searches == [
            (query_harness.index.collection, 5, PUBLIC, None)
        ]

    def test_when_top_k_omitted_then_default_and_cap_applied(
        self, query_harness: QueryHarness
    ) -> None:
        service = query_harness.variant(
            config=QueryConfig(retriever=RetrieverConfig(top_k=1, max_top_k=2))
        )

        assert len(service.search(search()).sources) == 1
        assert len(service.search(search(top_k=50)).sources) == 2
        assert [s[1] for s in query_harness.search_index.searches] == [1, 2]

    def test_when_filtered_by_asset_then_forwarded(self, query_harness: QueryHarness) -> None:
        query_harness.service.search(search(filters=models.SearchFilters(asset_version=ASSET)))

        assert query_harness.search_index.searches[-1][3] == ASSET

    @pytest.mark.parametrize(
        "question, top_k, message",
        [
            ("", None, "blank"),
            ("   ", None, "blank"),
            ("x" * 2001, None, "2000"),
            ("q", 0, "top_k"),
        ],
    )
    def test_when_request_invalid_then_invalid_query(
        self, query_harness: QueryHarness, question: str, top_k: int | None, message: str
    ) -> None:
        with pytest.raises(InvalidQueryError, match=message):
            query_harness.service.search(search(question, top_k=top_k))
        with pytest.raises(InvalidQueryError, match=message):
            query_harness.service.answer(answer(question, top_k=top_k))
        assert query_harness.embedder.questions == []

    def test_when_caller_has_no_entitlements_then_nothing_and_no_model_call(
        self, query_harness: QueryHarness
    ) -> None:
        anonymous = models.Caller(frozenset())

        found = query_harness.service.search(models.SearchQuery("q", anonymous))
        answered = query_harness.service.answer(models.AnswerQuery("q", anonymous))

        assert found.sources == ()
        assert (answered.refused, answered.sources, answered.citations) == (True, (), ())
        assert answered.answer == query_harness.config.refusal.message
        assert answered.usage == models.NO_USAGE
        assert query_harness.embedder.questions == []
        assert query_harness.generator.prompts == []

    def test_when_caller_entitled_differently_then_only_its_sources(
        self, query_harness: QueryHarness
    ) -> None:
        result = query_harness.service.search(models.SearchQuery("team only", models.Caller(TEAM)))

        assert [s.chunk_version for s in result.sources] == [ENTRIES[2].chunk_version]

    def test_when_asset_retired_then_its_sources_dropped(self, query_harness: QueryHarness) -> None:
        service = query_harness.variant(retired=FakeRetiredFilter(ASSET))

        assert service.search(search()).sources == ()

    def test_when_answered_then_prompt_lists_sources_and_citations_checked(
        self, query_harness: QueryHarness
    ) -> None:
        query_harness.generator.text = "At four seconds [2], see also [1] and [7]."

        result = query_harness.service.answer(answer())

        assert result.refused is False
        assert result.answer == "At four seconds [2], see also [1] and [7]."
        assert result.citations == (2, 1)
        assert result.invalid_citations == (7,)
        assert len(result.sources) == 2
        assert result.usage == models.Usage(100, 12)
        assert result.served == query_harness.served
        (prompt,) = query_harness.generator.prompts
        assert prompt[0].role == "system"
        assert "[1] clip.mp4 4.0-8.0s: chunk 1 text" in prompt[1].content
        assert "[2] clip.mp4 0.0-4.0s: chunk 0 text" in prompt[1].content
        assert prompt[1].content.endswith("Question: chunk 1 text")

    def test_when_best_score_below_threshold_then_refused_without_generator(
        self, query_harness: QueryHarness
    ) -> None:
        result = query_harness.service.answer(answer("something unrelated entirely"))

        assert result.refused is True
        assert result.answer == query_harness.config.refusal.message
        assert (result.citations, result.invalid_citations) == ((), ())
        assert all(s.score < query_harness.config.refusal.min_score for s in result.sources)
        assert result.usage == models.NO_USAGE
        assert query_harness.generator.prompts == []

    def test_when_generator_fails_then_error_propagates(self, query_harness: QueryHarness) -> None:
        service = query_harness.variant(generator=FakeAnswerGenerator(failure=ModelError("down")))

        with pytest.raises(ModelError, match="down"):
            service.answer(answer())


class TestGivenSceneApi:
    """Scene records and derivatives behind the fake scene-api."""

    RECORD = {
        "schema_version": 1,
        "asset_version": ASSET,
        "asset": {"entitlements": ["public", "team-platform"]},
        "scene": {"index": 3},
    }

    @pytest.fixture
    def query_harness(self, query_harness: QueryHarness) -> QueryHarness:
        query_harness.scenes.records[("sha256:set", 3)] = dict(self.RECORD)
        query_harness.scenes.derivatives["sha256:kf"] = models.DerivativeContent(
            "image/jpeg", b"\xff\xd8"
        )
        query_harness.scenes.derivatives["sha256:clip"] = models.DerivativeContent(
            "audio/wav", b"RIFF"
        )
        return query_harness

    def test_when_scene_visible_then_record_passed_through(
        self, query_harness: QueryHarness
    ) -> None:
        assert query_harness.service.get_scene("sha256:set", 3, PUBLIC_CALLER) == self.RECORD

    @pytest.mark.parametrize(
        "caller", [models.Caller(frozenset()), models.Caller(frozenset({"other"}))]
    )
    def test_when_scene_not_visible_then_not_found(
        self, query_harness: QueryHarness, caller: models.Caller
    ) -> None:
        with pytest.raises(NotFoundError, match="scene sha256:set/3"):
            query_harness.service.get_scene("sha256:set", 3, caller)

    def test_when_scene_absent_then_not_found(self, query_harness: QueryHarness) -> None:
        with pytest.raises(NotFoundError, match="scene sha256:set/4"):
            query_harness.service.get_scene("sha256:set", 4, PUBLIC_CALLER)

    def test_when_scene_asset_retired_then_not_found(self, query_harness: QueryHarness) -> None:
        service = query_harness.variant(retired=FakeRetiredFilter(ASSET))

        with pytest.raises(NotFoundError):
            service.get_scene("sha256:set", 3, PUBLIC_CALLER)

    def test_when_record_lacks_entitlements_then_scene_api_error(
        self, query_harness: QueryHarness
    ) -> None:
        query_harness.scenes.records[("sha256:set", 3)] = {"schema_version": 1}

        with pytest.raises(SceneApiError, match="entitlements"):
            query_harness.service.get_scene("sha256:set", 3, PUBLIC_CALLER)

    def test_when_keyframe_fetched_then_bytes(self, query_harness: QueryHarness) -> None:
        content = query_harness.service.get_keyframe("sha256:kf", PUBLIC_CALLER)

        assert content == models.DerivativeContent("image/jpeg", b"\xff\xd8")

    @pytest.mark.parametrize("digest", ["sha256:clip", "sha256:missing"])
    def test_when_not_a_keyframe_then_not_found(
        self, query_harness: QueryHarness, digest: str
    ) -> None:
        with pytest.raises(NotFoundError, match=f"keyframe {digest}"):
            query_harness.service.get_keyframe(digest, PUBLIC_CALLER)

    def test_when_caller_anonymous_then_keyframe_not_found_and_not_fetched(
        self, query_harness: QueryHarness
    ) -> None:
        with pytest.raises(NotFoundError, match="keyframe sha256:kf"):
            query_harness.service.get_keyframe("sha256:kf", models.ANONYMOUS)

        assert query_harness.scenes.calls == []
