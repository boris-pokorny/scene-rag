"""Extension points and store contracts of the enrichment step.

`Enricher` is the extension point; `DescriptionEnricher` and `TranscriptionEnricher` are the
reference implementations, over the `VisionModel` and `Transcriber` backend protocols.
`AssetVersionReader` and `SceneSetReader` are the step's read-only views of the Asset Registry
and the Scene Store. `EnrichmentStoreReader` and `EnrichmentStoreWriter` split the Enrichment
Store into queries and commands: the step depends on both, every other step and role of the
service may depend on the reader only. `DerivativeReader` yields the bytes a scene set
references. `SceneSetEnricher` is the per-scene-set unit of work the pass scheduler drives, and
`Dispatcher` the concurrency policy it runs under.
"""

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from scene_rag.extraction.enrichment import models


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


class AssetVersionReader(Protocol):
    """Queries over the Asset Registry, as far as enrichment needs them."""

    def get(self, version: str) -> models.AssetVersionView | None:
        """Return the version with the given content hash, live or retired."""
        ...

    def list_live_versions(self) -> frozenset[str]:
        """Return the content hashes of every live version."""
        ...


class SceneSetReader(Protocol):
    """Queries over the Scene Store, as far as enrichment needs them."""

    def get(self, version: str) -> models.SceneSetView | None:
        """Return the scene set with the given data version."""
        ...

    def list_summaries(self) -> Sequence[models.SceneSetSummary]:
        """Return a summary of every scene set, in version order."""
        ...


class EnrichmentStoreReader(Protocol):
    """Queries over the Enrichment Store; the only side consumers outside enrichment may use."""

    def get(self, version: str) -> models.Enrichment | None:
        """Return the record with the given data version."""
        ...

    def list_done(self, enricher: str, fingerprint: str) -> frozenset[tuple[str, int]]:
        """Return the ``(scene set version, scene index)`` pairs the enricher has a record for.

        Args:
            enricher: Name of the enricher.
            fingerprint: Fingerprint the records were written under.
        """
        ...


class EnrichmentStoreWriter(Protocol):
    """Commands over the Enrichment Store; each one is atomic in the backend."""

    def put(self, record: models.Enrichment) -> bool:
        """Insert a record.

        Returns:
            ``True`` when inserted, ``False`` when a record with that version already existed;
            the stored one is then left untouched, since equal inputs and fingerprint give
            equal output.
        """
        ...


class DerivativeUnavailableError(RuntimeError):
    """A derivative cannot be read as segmentation filed it: missing, or its bytes changed."""


class DerivativeReader(Protocol):
    """Reads the bytes a derivative reference points at."""

    def read(self, reference: models.DerivativeRefView) -> bytes:
        """Return the bytes of the derivative, verified against its digest.

        Raises:
            DerivativeUnavailableError: When the file cannot be read or its content hash is
                not the one the reference names.
        """
        ...


class ModelError(RuntimeError):
    """A model backend failed to answer."""


class VisionModel(Protocol):
    """A vision-language model: text about a set of images."""

    @property
    def model(self) -> models.ModelRef:
        """The model and its digest."""
        ...

    def describe(self, images: Sequence[bytes], prompt: str) -> str:
        """Answer ``prompt`` about ``images``.

        Raises:
            ModelError: When the backend cannot answer.
        """
        ...


@dataclasses.dataclass(frozen=True)
class TranscriptSegment:
    """A span of speech within an audio clip, timed from the start of the clip.

    Attributes:
        start_seconds: Start of the span within the clip.
        end_seconds: End of the span within the clip.
        text: The words recognised.
        confidence: Probability the backend assigns to the text, 0 to 1, when it reports one.
    """

    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None


@dataclasses.dataclass(frozen=True)
class Transcript:
    """What a `Transcriber` heard in one clip.

    Attributes:
        language: Detected or given language code; ``None`` when the backend reports none.
        segments: Timed spans in clip order; empty when no speech was found.
    """

    language: str | None
    segments: tuple[TranscriptSegment, ...]


class Transcriber(Protocol):
    """A speech-to-text model over one audio clip."""

    @property
    def model(self) -> models.ModelRef:
        """The model and its digest."""
        ...

    def transcribe(self, audio: bytes) -> Transcript:
        """Transcribe an audio file given as bytes (any container the backend decodes).

        Raises:
            ModelError: When the backend cannot decode or transcribe the audio.
        """
        ...


class Enricher(Protocol):
    """Produces semantic segments for one scene from the derivatives it needs."""

    @property
    def name(self) -> str:
        """Name keyed into every record the enricher produces."""
        ...

    @property
    def classification(self) -> str:
        """Data classification of the text the enricher produces."""
        ...

    @property
    def model(self) -> models.ModelRef:
        """The model behind the enricher."""
        ...

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the enricher: its hyperparameters, code and model digest."""
        ...

    def enrich(
        self, scene_set: models.SceneSetView, scene: models.SceneView
    ) -> models.EnricherOutput:
        """Enrich one scene.

        Returns:
            The segments and the derivatives read; empty segments when the scene offers
            nothing to the enricher.

        Raises:
            DerivativeUnavailableError: When a derivative the scene references cannot be read.
            ModelError: When the model backend fails.
        """
        ...


class VersionRetiredError(RuntimeError):
    """The asset version was retired, or its scene set vanished, during enrichment."""


class SceneSetEnricher(Protocol):
    """Runs the pending enrichers over the pending scenes of one scene set."""

    @property
    def fingerprints(self) -> Mapping[str, str]:
        """Fingerprint per enricher name, stamped on the records each one writes."""
        ...

    def enrich(
        self, summary: models.SceneSetSummary, pending: Mapping[str, frozenset[int]]
    ) -> models.Outcome:
        """Enrich the pending scenes of a scene set and store the records.

        A scene that cannot be enriched is skipped, not fatal: the remaining scenes and
        enrichers still run, and the records written stay, so the next pass derives only the
        skipped scenes as pending.

        Args:
            summary: The scene set to load.
            pending: Scene indices to enrich, per enricher name.

        Returns:
            ``"failed"`` when a backend or store failed on at least one scene, else
            ``"deferred"`` when a derivative of at least one scene could not be read as filed,
            else ``"enriched"`` when at least one record was written, else ``"duplicate"``
            (another worker wrote every pending record first).

        Raises:
            VersionRetiredError: When the asset version is retired, or the scene set is gone;
                records written before it stay.
        """
        ...
