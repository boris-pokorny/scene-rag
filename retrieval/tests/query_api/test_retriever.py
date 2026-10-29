"""Dense retrieval over the fake embedder and Search Index."""

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.retriever import DenseRetriever, source_of
from tests.support.query_fakes import (
    PUBLIC,
    FakeQueryEmbedder,
    FakeSearchReader,
    fake_vector,
    make_entry_view,
    make_index_view,
)

INDEX = make_index_view()
ENTRIES = [make_entry_view(i) for i in range(3)]


def retriever() -> tuple[DenseRetriever, FakeQueryEmbedder, FakeSearchReader]:
    embedder = FakeQueryEmbedder(INDEX)
    search_index = FakeSearchReader()
    search_index.seed(INDEX.collection, [(fake_vector(e.text), e) for e in ENTRIES])
    return DenseRetriever(embedder, search_index, INDEX), embedder, search_index


class TestGivenIndexedEntries:
    """Three public entries of one asset."""

    def test_when_retrieved_then_question_embedded_and_filter_forwarded(self) -> None:
        dense, embedder, search_index = retriever()
        caller = models.Caller(frozenset({"public", "x"}))

        sources = dense.retrieve("chunk 1 text", top_k=2, caller=caller, asset_version="a")

        assert embedder.questions == ["chunk 1 text"]
        assert search_index.searches == [(INDEX.collection, 2, frozenset({"public", "x"}), "a")]
        assert sources == []  # asset "a" holds nothing

    def test_when_retrieved_then_hits_become_sources_best_first(self) -> None:
        dense, _, _ = retriever()

        sources = dense.retrieve(
            "chunk 1 text", top_k=2, caller=models.Caller(PUBLIC), asset_version=None
        )

        assert len(sources) == 2
        assert sources[0].chunk_version == ENTRIES[1].chunk_version
        assert sources[0].score == 1.0
        assert sources[0] == source_of(models.Hit(ENTRIES[1], 1.0))
        assert sources[0].scene_indices == (1,)
        assert sources[0].sources == ENTRIES[1].sources

    def test_when_hit_converted_then_every_citation_field_carried(self) -> None:
        source = source_of(models.Hit(ENTRIES[2], 0.4))

        assert source == models.Source(
            chunk_version=ENTRIES[2].chunk_version,
            asset_version=ENTRIES[2].asset_version,
            scene_set_version=ENTRIES[2].scene_set_version,
            scene_indices=(2,),
            start_seconds=8.0,
            end_seconds=12.0,
            score=0.4,
            text="chunk 2 text",
            sources=ENTRIES[2].sources,
            entitlements=PUBLIC,
        )
