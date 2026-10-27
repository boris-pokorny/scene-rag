"""Extension points and store contracts of the ingestion step.

`MediaSource` is the connector extension point. `AssetRegistryReader` and
`AssetRegistryWriter` split the Asset Registry into queries and commands: the step depends on
both, every other step and role of the service may depend on the reader only.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import BinaryIO, Protocol

from scene_rag.extraction.ingestion import models


class DiscoveredMedia(Protocol):
    """One item of a source listing, readable on demand.

    Listing is cheap and must not read the bytes; `open` is called only when the item's
    reference does not match what the registry already holds.
    """

    @property
    def asset_id(self) -> str:
        """Stable identifier of the item within its source."""

    @property
    def uri(self) -> str:
        """Where the bytes can be read from."""

    @property
    def filename(self) -> str:
        """Display name of the item in its source."""

    @property
    def media_type(self) -> str:
        """MIME type the connector attributes to the item."""

    @property
    def size_bytes(self) -> int:
        """Size reported by the listing."""

    @property
    def modified_at(self) -> datetime:
        """Modification time reported by the listing, UTC, millisecond precision at most."""

    @property
    def entitlements(self) -> frozenset[str]:
        """Who may see the item according to the source."""

    def open(self) -> BinaryIO:
        """Open the bytes for reading.

        Returns:
            A binary stream positioned at the start; the caller closes it.

        Raises:
            OSError: When the bytes cannot be read.
        """
        ...

    def refresh(self) -> DiscoveredMedia:
        """Re-read the item's size and modification time from the source.

        Called after hashing, to detect an item that changed while its bytes were read.

        Returns:
            A new item describing the same asset as the source reports it now.

        Raises:
            OSError: When the item is gone or cannot be inspected.
        """
        ...


class MediaSource(Protocol):
    """A connector: the anti-corruption layer over one source system."""

    @property
    def source_id(self) -> str:
        """Identifier stamped on every source reference the connector produces."""

    def list(self) -> Iterable[DiscoveredMedia]:
        """Enumerate every item the source currently holds.

        Yields:
            One item per asset; an item missing from the listing is treated as deleted.

        Raises:
            OSError: When the source cannot be listed at all.
        """
        ...


class AssetRegistryReader(Protocol):
    """Queries over the Asset Registry; the only side consumers outside ingestion may use."""

    def get(self, version: str) -> models.AssetVersion | None:
        """Return the version with the given content hash, live or retired."""
        ...

    def list_by_source(self, source_id: str) -> Sequence[models.AssetVersion]:
        """Return every live version holding at least one reference from the source."""
        ...


class UnknownVersionError(LookupError):
    """A command named a version the registry does not hold."""


class AssetRegistryWriter(Protocol):
    """Commands over the Asset Registry; each one is atomic in the backend."""

    def register_version(self, record: models.AssetVersion) -> bool:
        """Insert a new version.

        Args:
            record: The version with its first source reference(s).

        Returns:
            ``True`` when inserted, ``False`` when the version already existed; the record
            is then left untouched and the caller adds its reference instead.
        """
        ...

    def add_source_reference(self, version: str, reference: models.SourceReference) -> None:
        """Add or refresh a reference on an existing version and clear its tombstone.

        Idempotent: a reference with the same key replaces the previous one.

        Raises:
            UnknownVersionError: When the version does not exist.
        """
        ...

    def remove_source_reference(
        self, version: str, source_id: str, asset_id: str, *, at: datetime
    ) -> models.AssetVersion:
        """Remove a reference; set ``retired_at`` to ``at`` when none remain.

        Idempotent: removing an absent reference changes nothing. An existing tombstone is
        kept.

        Returns:
            The record after the command.

        Raises:
            UnknownVersionError: When the version does not exist.
        """
        ...


class Clock(Protocol):
    """Source of the current time, so timestamps are deterministic in tests."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        ...
