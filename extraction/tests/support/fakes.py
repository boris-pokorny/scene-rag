"""Fakes and builders shared by the tests of the extraction service."""

import dataclasses
import hashlib
import io
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import BinaryIO

from scene_rag.extraction.ingestion import models
from scene_rag.extraction.ingestion.config import IngestionConfig
from scene_rag.extraction.ingestion.protocols import (
    AssetRegistryWriter,
    MediaSource,
    UnknownVersionError,
)
from scene_rag.extraction.ingestion.service import IngestionService

T0 = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)
# Listed items are a minute old by default: older than `IngestionConfig.min_age_seconds`, so
# the hashing path runs; a test that wants deferral dates its item at the clock's time.
AGED = timedelta(minutes=1)
SOURCE_ID = "local"
PUBLIC = frozenset({"public"})


def sha256_version(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class FakeClock:
    """Clock that returns a fixed time and advances only when told."""

    def __init__(self, start: datetime = T0) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current

    def tick(self, seconds: float = 60) -> datetime:
        self.current += timedelta(seconds=seconds)
        return self.current


@dataclasses.dataclass(frozen=True)
class FakeMedia:
    """DiscoveredMedia backed by bytes in memory.

    `declared_size` makes the listing lie about the length of `data`; `refreshed` is what
    `refresh` returns instead of the item itself, to simulate a file changing while hashed.
    """

    asset_id: str
    data: bytes
    filename: str = "clip.mp4"
    media_type: str = "video/mp4"
    modified_at: datetime = T0 - AGED
    entitlements: frozenset[str] = PUBLIC
    fail_open: bool = False
    read_sizes: list[int] | None = None
    declared_size: int | None = None
    refreshed: FakeMedia | None = None
    fail_refresh: bool = False

    @property
    def uri(self) -> str:
        return f"fake:///{self.asset_id}/{self.filename}"

    @property
    def size_bytes(self) -> int:
        return len(self.data) if self.declared_size is None else self.declared_size

    def open(self) -> BinaryIO:
        if self.fail_open:
            raise OSError(f"cannot read {self.uri}")
        if self.read_sizes is not None:
            return RecordingStream(self.data, self.read_sizes)
        return io.BytesIO(self.data)

    def refresh(self) -> FakeMedia:
        if self.fail_refresh:
            raise OSError(f"cannot inspect {self.uri}")
        return self if self.refreshed is None else self.refreshed


class RecordingStream(io.BytesIO):
    """BytesIO that records the size requested by every read."""

    def __init__(self, data: bytes, read_sizes: list[int]) -> None:
        super().__init__(data)
        self._read_sizes = read_sizes

    def read(self, size: int | None = -1, /) -> bytes:
        self._read_sizes.append(-1 if size is None else size)
        return super().read(size)


class FailingMediaSource:
    """MediaSource whose listing raises."""

    source_id = SOURCE_ID

    def list(self) -> Iterator[FakeMedia]:
        raise OSError("source down")


class FakeMediaSource:
    """MediaSource over a mutable list of FakeMedia items."""

    def __init__(self, items: list[FakeMedia] | None = None, source_id: str = SOURCE_ID) -> None:
        self.items = items if items is not None else []
        self.source_id = source_id

    def list(self) -> Iterator[FakeMedia]:
        yield from list(self.items)


class FakeAssetRegistry:
    """Asset Registry in a dictionary, with the command semantics of the MongoDB backend.

    `tests/ingestion/test_registry.py` runs the same cases against both, so the fake stays
    faithful.
    """

    def __init__(self) -> None:
        self._records: dict[str, models.AssetVersion] = {}

    def get(self, version: str) -> models.AssetVersion | None:
        return self._records.get(version)

    def snapshot(self) -> dict[str, models.AssetVersion]:
        """Return every record, live or retired, for whole-state assertions."""
        return dict(self._records)

    def list_by_source(self, source_id: str) -> Sequence[models.AssetVersion]:
        return [
            record
            for version, record in sorted(self._records.items())
            if record.retired_at is None
            and any(reference.source_id == source_id for reference in record.sources)
        ]

    def register_version(self, record: models.AssetVersion) -> bool:
        if record.version in self._records:
            return False
        self._records[record.version] = record
        return True

    def add_source_reference(self, version: str, reference: models.SourceReference) -> None:
        record = self._require(version)
        kept = tuple(source for source in record.sources if source.key != reference.key)
        self._records[version] = dataclasses.replace(
            record, sources=(*kept, reference), retired_at=None
        )

    def remove_source_reference(
        self, version: str, source_id: str, asset_id: str, *, at: datetime
    ) -> models.AssetVersion:
        record = self._require(version)
        kept = tuple(source for source in record.sources if source.key != (source_id, asset_id))
        retired_at = at if not kept and record.retired_at is None else record.retired_at
        updated = dataclasses.replace(record, sources=kept, retired_at=retired_at)
        self._records[version] = updated
        return updated

    def _require(self, version: str) -> models.AssetVersion:
        record = self._records.get(version)
        if record is None:
            raise UnknownVersionError(version)
        return record


class RacingWriter:
    """Writer whose first `register_version` loses a race: the version appears meanwhile."""

    def __init__(self, inner: FakeAssetRegistry) -> None:
        self._inner = inner
        self.raced = False

    def register_version(self, record: models.AssetVersion) -> bool:
        if not self.raced:
            self.raced = True
            rival = dataclasses.replace(
                record, sources=(make_reference("asset-00000000000000ff", source_id="rival"),)
            )
            self._inner.register_version(rival)
        return self._inner.register_version(record)

    def add_source_reference(self, version: str, reference: models.SourceReference) -> None:
        self._inner.add_source_reference(version, reference)

    def remove_source_reference(
        self, version: str, source_id: str, asset_id: str, *, at: datetime
    ) -> models.AssetVersion:
        return self._inner.remove_source_reference(version, source_id, asset_id, at=at)


class RecordingWriter:
    """AssetRegistryWriter that logs every command, delegating to a real writer."""

    def __init__(self, inner: AssetRegistryWriter, fail_on: str | None = None) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self.commands: list[tuple[str, ...]] = []

    def _record(self, *command: str) -> None:
        self.commands.append(command)
        if self._fail_on == command[0]:
            raise RuntimeError(f"injected failure in {command[0]}")

    def register_version(self, record: models.AssetVersion) -> bool:
        self._record("register_version", record.version)
        return self._inner.register_version(record)

    def add_source_reference(self, version: str, reference: models.SourceReference) -> None:
        self._record("add_source_reference", version, reference.asset_id)
        self._inner.add_source_reference(version, reference)

    def remove_source_reference(
        self, version: str, source_id: str, asset_id: str, *, at: datetime
    ) -> models.AssetVersion:
        self._record("remove_source_reference", version, asset_id)
        return self._inner.remove_source_reference(version, source_id, asset_id, at=at)


def make_reference(
    asset_id: str = "asset-0000000000000001",
    source_id: str = SOURCE_ID,
    size_bytes: int = 4,
    modified_at: datetime = T0,
    entitlements: frozenset[str] = PUBLIC,
    seen_at: datetime = T0,
) -> models.SourceReference:
    return models.SourceReference(
        source_id=source_id,
        asset_id=asset_id,
        uri=f"file:///media/{asset_id}.mp4",
        filename=f"{asset_id}.mp4",
        size_bytes=size_bytes,
        modified_at=modified_at,
        entitlements=entitlements,
        seen_at=seen_at,
    )


def make_record(
    version: str = sha256_version(b"abcd"),
    *references: models.SourceReference,
    retired_at: datetime | None = None,
) -> models.AssetVersion:
    sources = references or (make_reference(),)
    return models.AssetVersion(
        version=version,
        media_type="video/mp4",
        size_bytes=sources[0].size_bytes,
        sources=tuple(sources),
        first_seen_at=T0,
        lineage=models.Lineage(fingerprint="0123456789ab", version=version),
        retired_at=retired_at,
    )


@dataclasses.dataclass
class Harness:
    """A service wired to fakes, exposing them for assertions."""

    source: FakeMediaSource
    registry: FakeAssetRegistry
    writer: RecordingWriter
    clock: FakeClock
    service: IngestionService

    def variant(
        self,
        *,
        source: MediaSource | None = None,
        writer: AssetRegistryWriter | None = None,
        config: IngestionConfig | None = None,
    ) -> IngestionService:
        """Build a second service over the same fakes with one collaborator replaced."""
        return IngestionService(
            source=source or self.source,
            reader=self.registry,
            writer=writer or self.writer,
            clock=self.clock,
            config=config or IngestionConfig(),
        )


def make_harness(*items: FakeMedia) -> Harness:
    """Wire a service to fresh fakes over the given items; `Harness.variant` swaps parts."""
    source = FakeMediaSource(list(items))
    registry = FakeAssetRegistry()
    writer = RecordingWriter(registry)
    clock = FakeClock()
    service = IngestionService(
        source=source, reader=registry, writer=writer, clock=clock, config=IngestionConfig()
    )
    return Harness(source, registry, writer, clock, service)
