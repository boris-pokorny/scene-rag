"""Domain model of the ingestion step: asset versions, source references, lineage."""

import dataclasses
from datetime import datetime
from typing import Final

SCHEMA_VERSION: Final = 1
STEP_NAME: Final = "ingestion"


def truncate_to_millisecond(moment: datetime) -> datetime:
    """Return ``moment`` with its microseconds truncated to the millisecond.

    The registry stores BSON dates, which hold milliseconds. Every timestamp the step produces
    or compares is truncated first, so a round-tripped value equals a fresh one.
    """
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


@dataclasses.dataclass(frozen=True)
class SourceReference:
    """One source system's reference to the bytes of an asset version.

    Attributes:
        source_id: Identifier of the connector that found the item.
        asset_id: Stable identifier of the item within its source; survives content changes.
        uri: Where the connector read the bytes from.
        filename: Display name of the item in its source.
        size_bytes: Size reported by the source listing.
        modified_at: Modification time reported by the source listing, in UTC.
        entitlements: Who may see the item according to the source.
        seen_at: When ingestion last wrote the reference (registered or refreshed), in UTC.
            An item whose listing matches its reference is not rewritten, so this is not
            a heartbeat.
    """

    source_id: str
    asset_id: str
    uri: str
    filename: str
    size_bytes: int
    modified_at: datetime
    entitlements: frozenset[str]
    seen_at: datetime

    @property
    def key(self) -> tuple[str, str]:
        """Return the pair that identifies the reference across versions."""
        return (self.source_id, self.asset_id)


@dataclasses.dataclass(frozen=True)
class Lineage:
    """How a record was produced.

    Attributes:
        fingerprint: Fingerprint of the step that wrote the record.
        version: Data version of the record. For ingestion this is the content hash itself,
            the root of the Merkle DAG.
        step: Name of the producing step.
        inputs: Data versions read to produce the record; ingestion reads only source bytes.
        image: Digest of the service image, when known.
    """

    fingerprint: str
    version: str
    step: str = STEP_NAME
    inputs: tuple[str, ...] = ()
    image: str | None = None


@dataclasses.dataclass(frozen=True)
class AssetVersion:
    """An immutable asset version: one set of bytes, however many sources hold it.

    Attributes:
        version: Content hash as ``<algorithm>:<hex digest>``.
        media_type: MIME type of the bytes.
        size_bytes: Size of the bytes.
        sources: Every source reference currently holding these bytes.
        first_seen_at: When the version was first registered, in UTC.
        lineage: Provenance of the record.
        retired_at: Tombstone set when the last source reference was removed; ``None``
            while the version is live.
        schema_version: Version of the stored record's schema.
    """

    version: str
    media_type: str
    size_bytes: int
    sources: tuple[SourceReference, ...]
    first_seen_at: datetime
    lineage: Lineage
    retired_at: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def entitlements(self) -> frozenset[str]:
        """Return the union of the entitlements of every source reference."""
        return frozenset().union(*(source.entitlements for source in self.sources))

    @property
    def is_live(self) -> bool:
        """Return whether at least one source still holds the version and it is not retired."""
        return bool(self.sources) and self.retired_at is None


@dataclasses.dataclass(frozen=True)
class IngestionReport:
    """Counters of one ingestion pass over one source.

    Attributes:
        registered: New source references written, whether to a new or an existing version.
        unchanged: Listed items whose key already held the same bytes. Items whose listing
            matched their reference were not re-hashed (and were refreshed in place when
            only their metadata differed); the rest were hashed to the version already held.
        superseded: References removed because their item now hashes to another version.
        retired: References removed because their item left the source listing.
        quarantined: Listed items that failed validation; not registered, and any earlier
            reference of theirs is retired as if the item had left the listing.
        deferred: Listed items left for the next pass because they were modified too recently
            or changed while being hashed; any earlier reference of theirs is kept.
        failed: Listed items that raised; the pass continued past them.
    """

    registered: int = 0
    unchanged: int = 0
    superseded: int = 0
    retired: int = 0
    quarantined: int = 0
    deferred: int = 0
    failed: int = 0

    def __str__(self) -> str:
        """Render the counters as ``name=value`` pairs for logs."""
        return " ".join(
            f"{field.name}={getattr(self, field.name)}" for field in dataclasses.fields(self)
        )
