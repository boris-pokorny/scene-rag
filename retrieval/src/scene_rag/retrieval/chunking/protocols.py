"""Extension points and store contracts of the chunking step.

`Chunker` is the extension point; `PerSceneChunker` is the reference implementation.
`SceneRecordReader` is the step's read-only view of the scene-api of the extraction service.
`ChunkStoreReader` and `ChunkStoreWriter` split the Chunk Store into queries and commands: the
step depends on both, every other step and role of the service may depend on the reader only.
`SceneSetChunker` is the per-scene-set unit of work the pass scheduler drives, and
`Dispatcher` the concurrency policy it runs under.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from typing import Protocol

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.report import Outcome


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


class SceneApiError(RuntimeError):
    """The scene-api could not be reached, refused a request, or answered something unusable."""


class SceneRecordReader(Protocol):
    """Queries over the scene-api, as far as chunking needs them; pagination is hidden."""

    def list_scene_sets(self, query: models.SceneSetQuery) -> Iterator[models.SceneSetSummaryView]:
        """Yield the scene sets of live asset versions matching the query, in version order.

        Raises:
            SceneApiError: When the API cannot be read.
        """
        ...

    def list_scene_records(
        self, scene_set_version: str, pins: Sequence[models.EnrichmentPin]
    ) -> Sequence[models.SceneRecordView]:
        """Return every scene record of a scene set, in index order.

        The records carry the pinned enrichments only, or every enrichment without a pin.

        Raises:
            SceneApiError: When the API cannot be read, or the scene set is gone.
        """
        ...

    def get_asset_version(self, version: str) -> models.AssetVersionView | None:
        """Return the asset version, live or retired; ``None`` when the registry has none.

        Raises:
            SceneApiError: When the API cannot be read.
        """
        ...


class ChunkStoreReader(Protocol):
    """Queries over the Chunk Store; the only side consumers outside chunking may use."""

    def get_chunk(self, version: str) -> models.Chunk | None:
        """Return the chunk with the given data version."""
        ...

    def get_chunk_set(self, version: str) -> models.ChunkSet | None:
        """Return the chunk set with the given data version."""
        ...

    def list_done(self, fingerprint: str) -> frozenset[str]:
        """Return the scene set versions that have a chunk set under the fingerprint.

        The whole set is materialised: memory grows with the number of scene sets, which is
        acceptable for chunk sets (one per scene set) and must not be copied for chunks.
        """
        ...


class ChunkStoreWriter(Protocol):
    """Commands over the Chunk Store; each one is atomic in the backend."""

    def put_chunk(self, chunk: models.Chunk) -> bool:
        """Insert a chunk.

        Returns:
            ``True`` when inserted, ``False`` when a chunk with that version already existed;
            the stored one is then left untouched, since equal inputs and fingerprint give
            equal output.
        """
        ...

    def put_chunk_set(self, chunk_set: models.ChunkSet) -> bool:
        """Insert a chunk set; ``False`` when its version already existed."""
        ...


class Chunker(Protocol):
    """Cuts the scene records of one scene set into retrieval units."""

    @property
    def name(self) -> str:
        """Name of the chunking strategy, for logs."""
        ...

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the chunker: the step's hyperparameters and its code."""
        ...

    def chunk(self, records: Sequence[models.SceneRecordView]) -> tuple[models.ChunkDraft, ...]:
        """Chunk the records of one scene set, given in index order.

        Returns:
            The drafts in the order they are to be numbered; empty when no scene has text.
        """
        ...


class VersionRetiredError(RuntimeError):
    """The asset version was retired during chunking."""


class SceneSetChunker(Protocol):
    """Chunks one scene set and stores the result."""

    @property
    def fingerprint(self) -> str:
        """Fingerprint stamped on every record written."""
        ...

    def chunk(self, summary: models.SceneSetSummaryView) -> Outcome:
        """Chunk one scene set and store its chunks and chunk set.

        Returns:
            ``"chunked"`` when the chunk set was written, ``"duplicate"`` when another worker
            wrote it first, ``"deferred"`` when the records were not complete under the pins
            at fetch time.

        Raises:
            VersionRetiredError: When the asset version is retired; chunks written before
                it stay, the chunk set is not written.
            SceneApiError: When the scene-api cannot be read.
        """
        ...
