"""The null reranker."""

from scene_rag.retrieval.query_api.reranker import NoReranker
from tests.support.query_fakes import make_entry_view, make_source


class TestGivenSources:
    """Sources in retrieval order."""

    def test_when_reranked_then_unchanged(self) -> None:
        sources = [make_source(make_entry_view(i), 1.0 - i / 10) for i in range(3)]

        assert list(NoReranker().rerank("q", sources)) == sources
        assert NoReranker().rerank("q", []) == []
