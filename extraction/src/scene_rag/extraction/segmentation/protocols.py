"""Extension points and store contracts of the segmentation step.

`SegmentExtractor`, `KeyframeRenderer` and `AudioClipper` are the extension points; the
default backends are `scenedetect_backend` and `ffmpeg_backend`. `AssetVersionReader` is the
step's read-only view of the Asset Registry. `SceneStoreReader` and `SceneStoreWriter` split
the Scene Store into queries and commands: the step depends on both, every other step and role
of the service may depend on the reader only. `DerivativeStore` files rendered media by content.
`VersionSegmenter` is the per-version unit of work the pass scheduler drives, and `Dispatcher`
the concurrency policy it, and the backends, run their loops through.
"""

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from scene_rag.extraction.segmentation import models


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


class AssetVersionReader(Protocol):
    """Queries over the Asset Registry, as far as segmentation needs them."""

    def get(self, version: str) -> models.AssetVersionView | None:
        """Return the version with the given content hash, live or retired."""
        ...

    def list_live(self) -> Sequence[models.AssetVersionView]:
        """Return every live version, in version order."""
        ...


class SceneStoreReader(Protocol):
    """Queries over the Scene Store; the only side consumers outside segmentation may use."""

    def get(self, version: str) -> models.SceneSet | None:
        """Return the scene set with the given data version."""
        ...

    def list_asset_versions(self, fingerprint: str) -> frozenset[str]:
        """Return the asset versions that have a scene set under the fingerprint."""
        ...


class SceneStoreWriter(Protocol):
    """Commands over the Scene Store; each one is atomic in the backend."""

    def put(self, scene_set: models.SceneSet) -> bool:
        """Insert a scene set.

        Returns:
            ``True`` when inserted, ``False`` when a set with that version already existed;
            the stored one is then left untouched, since equal inputs and fingerprint give
            equal output.
        """
        ...


class DerivativeStore(Protocol):
    """The Derivative Store: files are addressed by their content, so writes are idempotent."""

    def put(self, path: Path, media_type: str) -> models.DerivativeRef:
        """File the bytes at ``path`` under their content hash.

        Args:
            path: A file the caller owns; the store may move it.
            media_type: MIME type to record for it.

        Returns:
            Where the bytes now live; the same reference for the same bytes, whenever called.

        Raises:
            OSError: When the file cannot be read or filed.
            ValueError: When the media type is not one the store files.
        """
        ...


class MediaUnavailableError(RuntimeError):
    """No source reference of a version yields its bytes as ingestion recorded them."""


class MediaLocator(Protocol):
    """Finds the bytes of an asset version on the local filesystem."""

    def locate(self, record: models.AssetVersionView) -> Path:
        """Return a local path holding exactly the bytes the version names.

        Raises:
            MediaUnavailableError: When no source reference can be read, or the file behind
                one has changed since ingestion hashed it.
        """
        ...


class VideoAnalysisError(RuntimeError):
    """A video could not be opened or analysed."""


@dataclasses.dataclass(frozen=True)
class Analysis:
    """Result of one analysis pass over a video.

    Attributes:
        media: Properties of the video stream.
        segments: Ordered segments tiling the stream, at least one.
    """

    media: models.MediaInfo
    segments: tuple[models.Segment, ...]


class SegmentExtractor(Protocol):
    """Cuts a video into segments: scene cuts by default; chapters or fixed windows elsewhere."""

    def extract(self, path: Path) -> Analysis:
        """Analyse the video at ``path``.

        Raises:
            VideoAnalysisError: When the file cannot be opened or decoded as video.
        """
        ...


@dataclasses.dataclass(frozen=True)
class RenderedFrame:
    """A frame written to disk by a `KeyframeRenderer`.

    Attributes:
        path: The image file.
        frame: Index of the frame in the stream.
        seconds: Position of the frame in the stream.
    """

    path: Path
    frame: int
    seconds: float


class KeyframeRenderer(Protocol):
    """Renders representative frames of each segment as JPEG files."""

    def render(
        self, path: Path, segments: Sequence[models.Segment], output_dir: Path
    ) -> Mapping[int, Sequence[RenderedFrame]]:
        """Write keyframes of every segment into ``output_dir``.

        Returns:
            Frames per segment index, in stream order, without duplicate frames; a segment
            may be absent when nothing could be rendered for it.

        Raises:
            VideoAnalysisError: When the file cannot be opened or decoded as video.
        """
        ...


class AudioExtractionError(RuntimeError):
    """Audio could not be extracted from a media file."""


class AudioClipper(Protocol):
    """Renders the audio of each segment as a WAV file."""

    def clip(
        self, path: Path, segments: Sequence[models.Segment], output_dir: Path
    ) -> Mapping[int, Path]:
        """Write one clip per segment into ``output_dir``.

        Returns:
            Clip file per segment index; empty when the asset has no audio stream. Empty
            segments, and segments the audio stream does not reach, get no clip.

        Raises:
            AudioExtractionError: When the tool fails on a readable file.
        """
        ...


class Clock(Protocol):
    """Source of the current time, so timestamps are deterministic in tests."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        ...


class VersionRetiredError(RuntimeError):
    """The version was retired while it was being segmented; nothing was written."""


class VersionSegmenter(Protocol):
    """Turns one live asset version into one scene set under a fixed fingerprint."""

    @property
    def fingerprint(self) -> str:
        """Fingerprint stamped on the lineage of every scene set written."""
        ...

    def segment(self, record: models.AssetVersionView) -> Literal["segmented", "duplicate"]:
        """Cut, render, clip and store the version.

        Returns:
            ``"segmented"`` when the scene set was written, ``"duplicate"`` when another worker
            wrote it first.

        Raises:
            MediaUnavailableError: When the version's bytes cannot be read as ingestion
                recorded them.
            VersionRetiredError: When the version was retired before the write.
            Exception: Any failure of a backend or store; the caller decides how to count it.
        """
        ...
