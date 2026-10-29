"""Extension points and store contracts of the embedding step.

`EmbeddingModel` is the extension point; `OllamaEmbeddingModel` is the reference
implementation. `ChunkReader` is the step's read-only view of the Chunk Store, which the
chunking step writes. `SearchIndexReader` and `SearchIndexWriter` split the Search Index
into queries and commands, `IndexStoreReader` and `IndexStoreWriter` do the same for the
index ledger (the index sets and the indexes); the step depends on both sides, every other
step and role of the service may depend on the readers only. `ChunkSetEmbedder` is the
per-chunk-set unit of work the pass scheduler drives, and `Dispatcher` the concurrency policy
it runs under.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from typing import Protocol

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.report import Outcome


class Dispatcher(Protocol):
    """Applies a function to every item of a batch under some concurrency policy.

    Implementations differ only in *when* the calls run (one after another, on a thread pool,
    elsewhere); every one yields the results in the order of the items, so what a caller
    builds from them never depends on the policy.
    """

    def map[T, R](self, function: Callable[[T], R], items: Iterable[T]) -> Iterator[R]:
        """Apply ``function`` to each item.

        Returns:
            The results in the order of ``items``. An exception raised by ``function``
            propagates from the iteration, at the position of the item that raised; calls
            already running finish before it propagates; items not yet started may be skipped.
        """
        ...


class Clock(Protocol):
    """Source of the current time, so timestamps are deterministic in tests."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        ...


class ModelError(RuntimeError):
    """The embedding model could not be reached, refused a request, or answered unusably."""


class SearchIndexError(RuntimeError):
    """The Search Index could not be reached, refused a request, or answered unusably."""


class IndexMismatchError(SearchIndexError):
    """A collection or index record exists under the fingerprint but was built differently."""


class ChunkReader(Protocol):
    """Queries over the Chunk Store, as far as embedding needs them."""

    def list_chunk_sets(self, chunking_fingerprint: str | None) -> Iterator[models.ChunkSetView]:
        """Yield the chunk sets written under the fingerprint, every one when ``None``.

        The order is stable across calls (by version), so a pass is reproducible.
        """
        ...

    def list_chunks(
        self, scene_set_version: str, chunking_fingerprint: str
    ) -> Sequence[models.ChunkView]:
        """Return the chunks of one scene set under one chunking fingerprint, in index order."""
        ...


class EmbeddingModel(Protocol):
    """Turns texts into vectors; the model behind the gateway."""

    @property
    def name(self) -> str:
        """Name of the backend, for logs."""
        ...

    @property
    def model(self) -> models.ModelRef:
        """The model and its digest."""
        ...

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the step: its hyperparameters, its code and the model digest."""
        ...

    @property
    def dimensions(self) -> int:
        """Length of every vector the model returns.

        Raises:
            ModelError: When the backend must be asked and cannot answer.
        """
        ...

    def embed(self, texts: Sequence[str]) -> tuple[models.Vector, ...]:
        """Embed the texts, one vector each, in the order given.

        Raises:
            ModelError: When the backend cannot answer, or answers the wrong number of vectors.
        """
        ...


class SearchIndexWriter(Protocol):
    """Commands over the Search Index."""

    def ensure_collection(self, name: str, dimensions: int, distance: str) -> None:
        """Create the collection when absent; verify it when present.

        Raises:
            IndexMismatchError: When the collection exists with other dimensions or distance.
            SearchIndexError: When the backend cannot be reached or refuses.
        """
        ...

    def upsert(self, collection: str, points: Sequence[models.Point]) -> None:
        """Write the points, replacing any with the same id; durable when it returns.

        Raises:
            SearchIndexError: When the backend cannot be reached or refuses.
        """
        ...


class SearchIndexReader(Protocol):
    """Queries over the Search Index, as far as the step and its checks need them."""

    def describe(self, collection: str) -> models.CollectionInfo | None:
        """Return the collection's dimensions, distance and size; ``None`` when absent.

        Raises:
            SearchIndexError: When the backend cannot be reached or refuses.
        """
        ...

    def count(self, collection: str, *, chunk_set_version: str | None = None) -> int:
        """Count the points of the collection, or of one chunk set within it.

        Raises:
            SearchIndexError: When the backend cannot be reached or refuses.
        """
        ...

    def get_entries(self, collection: str, point_ids: Sequence[str]) -> Sequence[models.IndexEntry]:
        """Return the entries of the given points, in the order asked, skipping absent ones.

        Raises:
            SearchIndexError: When the backend cannot be reached or refuses.
        """
        ...


class SearchIndex(SearchIndexReader, SearchIndexWriter, Protocol):
    """Both sides of the Search Index, as a backend provides them."""


class IndexStoreReader(Protocol):
    """Queries over the index ledger; the only side consumers outside embedding may use."""

    def get_index_set(self, version: str) -> models.IndexSet | None:
        """Return the index set with the given data version."""
        ...

    def get_index(self, fingerprint: str) -> models.Index | None:
        """Return the index built under the fingerprint."""
        ...

    def list_done(self, fingerprint: str) -> frozenset[str]:
        """Return the chunk set versions that have an index set under the fingerprint.

        The whole set is materialised: memory grows with the number of chunk sets (one per
        scene set), which is acceptable.
        """
        ...


class IndexStoreWriter(Protocol):
    """Commands over the index ledger; each one is atomic in the backend."""

    def put_index_set(self, index_set: models.IndexSet) -> bool:
        """Insert an index set.

        Returns:
            ``True`` when inserted, ``False`` when one with that version already existed;
            the stored one is then left untouched.
        """
        ...

    def put_index(self, index: models.Index) -> bool:
        """Insert an index record; ``False`` when its fingerprint already has one."""
        ...


class ChunkSetEmbedder(Protocol):
    """Embeds one chunk set and stores the result."""

    @property
    def fingerprint(self) -> str:
        """Fingerprint stamped on every record written."""
        ...

    def embed(self, chunk_set: models.ChunkSetView) -> Outcome:
        """Embed one chunk set, write its points and its index set.

        Returns:
            ``"indexed"`` when the index set was written, ``"duplicate"`` when another worker
            wrote it first, ``"deferred"`` when the chunks were not all there at fetch time.

        Raises:
            ModelError: When the model cannot embed.
            SearchIndexError: When the index cannot be written.
        """
        ...
