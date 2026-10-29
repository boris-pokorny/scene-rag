"""Dense retrieval: the question's vector against the served index, filtered before ranking.

`DenseRetriever` is the reference `Retriever`. Part of the ``query_config`` id:
how a hit becomes a source decides what is answered and cited.
"""

from collections.abc import Sequence

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.protocols import QueryEmbedder, SearchIndexReader


def source_of(hit: models.Hit) -> models.Source:
    """Turn a ranked point into a citable source."""
    entry = hit.entry
    return models.Source(
        chunk_version=entry.chunk_version,
        asset_version=entry.asset_version,
        scene_set_version=entry.scene_set_version,
        scene_indices=entry.scene_indices,
        start_seconds=entry.start_seconds,
        end_seconds=entry.end_seconds,
        score=hit.score,
        text=entry.text,
        sources=entry.sources,
        entitlements=entry.entitlements,
    )


class DenseRetriever:
    """`Retriever` over one embedder and one collection of the Search Index."""

    def __init__(
        self, embedder: QueryEmbedder, search_index: SearchIndexReader, index: models.IndexView
    ) -> None:
        """Bind to the served index.

        Args:
            embedder: Embeds questions the way the index was built.
            search_index: The Search Index backend.
            index: The index served; names the collection.
        """
        self._embedder = embedder
        self._search_index = search_index
        self._index = index

    def retrieve(
        self, question: str, *, top_k: int, caller: models.Caller, asset_version: str | None
    ) -> Sequence[models.Source]:
        """Embed the question and return the closest visible spans; see `Retriever`."""
        vector = self._embedder.embed(question)
        hits = self._search_index.search(
            self._index.collection,
            vector,
            limit=top_k,
            entitlements=caller.entitlements,
            asset_version=asset_version,
        )
        return [source_of(hit) for hit in hits]
