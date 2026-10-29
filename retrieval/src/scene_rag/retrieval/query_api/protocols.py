"""Ports of the query-api role, and the errors crossing them.

Every policy the role composes hides behind one of these: the retriever, the reranker, the
refusal, the generator, the read models. `bootstrap` picks the implementations; nothing else
names one.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from scene_rag.retrieval.query_api import models

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]


class InvalidQueryError(ValueError):
    """A request cannot be served as asked."""


class NotFoundError(LookupError):
    """A named resource does not exist, or is not visible to the caller.

    Attributes:
        kind: What was looked for.
        key: Its identifier.
    """

    def __init__(self, kind: str, key: str) -> None:
        """Name what was looked for."""
        super().__init__(f"no {kind} {key}")
        self.kind = kind
        self.key = key


class IndexUnavailableError(RuntimeError):
    """The index ledger or the Search Index cannot be read, or answers something unusable."""


class ModelError(RuntimeError):
    """The model backend cannot be reached, lacks the model or answers something unusable."""


class SceneApiError(RuntimeError):
    """The scene-api cannot be reached or answers something unusable."""


class IndexLedgerReader(Protocol):
    """Queries over the index ledger the embedding step writes."""

    def get_index(self, fingerprint: str) -> models.IndexView | None:
        """Return the index built under the embedding fingerprint, or ``None``.

        Raises:
            IndexUnavailableError: When the ledger cannot be read.
        """
        ...

    def newest_index(self) -> models.IndexView | None:
        """Return the most recently prepared index, or ``None`` when the ledger is empty.

        Raises:
            IndexUnavailableError: When the ledger cannot be read.
        """
        ...


class SearchIndexReader(Protocol):
    """Queries over the Search Index the embedding step writes."""

    def version(self) -> str:
        """Return the backend version; the connectivity check at startup.

        Raises:
            IndexUnavailableError: When the backend does not answer.
        """
        ...

    def describe(self, collection: str) -> models.CollectionInfo | None:
        """Return the collection's dimensions, distance and size; ``None`` when absent.

        Raises:
            IndexUnavailableError: When the backend does not answer or answers unusably.
        """
        ...

    def search(
        self,
        collection: str,
        vector: models.Vector,
        *,
        limit: int,
        entitlements: frozenset[str],
        asset_version: str | None,
    ) -> Sequence[models.Hit]:
        """Return the closest points the entitlements may see, best first.

        The entitlement filter is applied before ranking: a point is a candidate only when
        its entitlements share a member with ``entitlements``; empty matches nothing.

        Raises:
            IndexUnavailableError: When the backend does not answer or answers unusably.
        """
        ...


class QueryEmbedder(Protocol):
    """Embeds a question the way the served index was built."""

    @property
    def model(self) -> models.ModelRef:
        """Return the embedding model and its digest."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the length of the vectors."""
        ...

    def embed(self, question: str) -> models.Vector:
        """Return the question's vector, prefix applied.

        Raises:
            ModelError: When the backend fails or answers the wrong shape.
        """
        ...


class Retriever(Protocol):
    """Finds the spans relevant to a question; the `Retriever` extension point."""

    def retrieve(
        self, question: str, *, top_k: int, caller: models.Caller, asset_version: str | None
    ) -> Sequence[models.Source]:
        """Return up to ``top_k`` spans the caller may see, best first.

        Raises:
            ModelError: When the question cannot be embedded.
            IndexUnavailableError: When the index cannot be searched.
        """
        ...


class Reranker(Protocol):
    """Reorders retrieved spans for a question."""

    def rerank(self, question: str, sources: Sequence[models.Source]) -> Sequence[models.Source]:
        """Return the spans in the order the answer should see them."""
        ...


class RefusalPolicy(Protocol):
    """Decides whether a question is answered at all."""

    def should_refuse(self, sources: Sequence[models.Source]) -> bool:
        """Return whether the spans are too weak to answer from."""
        ...


class AnswerGenerator(Protocol):
    """Writes an answer over a prompt; the `AnswerGenerator` extension point."""

    @property
    def model(self) -> models.ModelRef:
        """Return the generator model and its digest."""
        ...

    def generate(self, messages: Sequence[models.Message]) -> models.Generation:
        """Return the answer to the chat prompt.

        Raises:
            ModelError: When the backend fails or answers the wrong shape.
        """
        ...


class SceneReader(Protocol):
    """Read model over the scene-api of the extraction service."""

    def health(self) -> None:
        """Check that the API answers; the connectivity check at startup.

        Raises:
            SceneApiError: When it does not.
        """
        ...

    def get_scene_record(self, scene_set_version: str, index: int) -> Json | None:
        """Return the published scene record, as is, or ``None`` when there is none.

        Raises:
            SceneApiError: When the API fails or answers a schema this role cannot read.
        """
        ...

    def get_derivative(self, digest: str) -> models.DerivativeContent | None:
        """Return the bytes of a derivative, or ``None`` when there is none.

        Raises:
            SceneApiError: When the API fails.
        """
        ...


class RetiredFilter(Protocol):
    """Blocks asset versions retired since the index was written."""

    def is_retired(self, asset_version: str) -> bool:
        """Return whether the asset version must not be served."""
        ...


class QueryService(Protocol):
    """The use cases of the role; what every adapter delegates to."""

    @property
    def served(self) -> models.Served:
        """Return what this process serves: query config and index."""
        ...

    def search(self, query: models.SearchQuery) -> models.SearchResult:
        """Return the spans relevant to the question that the caller may see.

        Raises:
            InvalidQueryError: When the question or the count cannot be used.
            ModelError: When the question cannot be embedded.
            IndexUnavailableError: When the index cannot be searched.
        """
        ...

    def answer(self, query: models.AnswerQuery) -> models.AnswerResult:
        """Return an answer written over the spans, or a refusal.

        Raises:
            InvalidQueryError: When the question or the count cannot be used.
            ModelError: When a model fails.
            IndexUnavailableError: When the index cannot be searched.
        """
        ...

    def get_scene(self, scene_set_version: str, index: int, caller: models.Caller) -> Json:
        """Return the scene record, when the caller may see its asset.

        Raises:
            NotFoundError: When there is no such record, or the caller may not see it.
            SceneApiError: When the scene-api fails.
        """
        ...

    def get_keyframe(self, digest: str, caller: models.Caller) -> models.DerivativeContent:
        """Return the bytes of a keyframe.

        Raises:
            NotFoundError: When there is no such derivative, it is not a keyframe, or the
                caller has no entitlements.
            SceneApiError: When the scene-api fails.
        """
        ...
