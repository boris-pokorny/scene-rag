"""Reranking of retrieved sources; the role ships the null object.

`NoReranker` keeps the retriever's order: the reference behaviour, and the extension point
a cross-encoder or a fusion of several retrievers plugs into, should hybrid retrieval be
added. Part of the ``query_config`` id.
"""

from collections.abc import Sequence

from scene_rag.retrieval.query_api import models


class NoReranker:
    """`Reranker` that changes nothing."""

    def rerank(self, question: str, sources: Sequence[models.Source]) -> Sequence[models.Source]:
        """Return the sources as they came."""
        return list(sources)
