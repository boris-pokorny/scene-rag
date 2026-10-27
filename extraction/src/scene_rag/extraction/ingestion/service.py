"""The ingestion pass: reconcile one source's listing with the Asset Registry."""

import dataclasses
import hashlib
import logging
from collections import defaultdict
from datetime import datetime
from typing import Final

from scene_rag.extraction.ingestion import models
from scene_rag.extraction.ingestion.config import IngestionConfig
from scene_rag.extraction.ingestion.fingerprint import compute_fingerprint
from scene_rag.extraction.ingestion.protocols import (
    AssetRegistryReader,
    AssetRegistryWriter,
    Clock,
    DiscoveredMedia,
    MediaSource,
)

logger = logging.getLogger(__name__)

_READ_CHUNK_BYTES: Final = 1 << 20

type _Holders = list[tuple[models.AssetVersion, models.SourceReference]]


class IngestionService:
    """Register what a source holds, supersede what changed, retire what disappeared.

    One pass lists the source once and the registry once, then issues one command per
    change. Items are never deleted from the registry: a version nobody holds any more
    gets a tombstone. New references are written before old ones are
    removed, so an asset never vanishes while it is re-registered.

    An item whose size and modification time match its stored reference is not re-hashed;
    when only its metadata (uri, filename, entitlements) differs, the reference is refreshed
    in place so that, for example, a change of entitlements reaches the registry.

    Items failing validation are quarantined: logged as warnings, counted, never hashed or
    registered, and treated as absent so that any earlier reference of theirs is retired.
    """

    def __init__(
        self,
        *,
        source: MediaSource,
        reader: AssetRegistryReader,
        writer: AssetRegistryWriter,
        clock: Clock,
        config: IngestionConfig,
    ) -> None:
        """Compose the pass from its collaborators; `bootstrap` is the only caller."""
        self._source = source
        self._reader = reader
        self._writer = writer
        self._clock = clock
        self._config = config
        self._fingerprint = compute_fingerprint(config)

    @property
    def fingerprint(self) -> str:
        """Fingerprint stamped on the lineage of every version this service registers."""
        return self._fingerprint

    def run_once(self) -> models.IngestionReport:
        """Run one pass over the source.

        Returns:
            Counters of what the pass did. A failing item is counted and skipped; a failing
            listing propagates, since nothing can be decided without it.
        """
        now = self._clock.now()
        source_id = self._source.source_id
        held: defaultdict[tuple[str, str], _Holders] = defaultdict(list)
        for record in self._reader.list_by_source(source_id):
            for reference in record.sources:
                if reference.source_id == source_id:
                    held[reference.key].append((record, reference))

        report = models.IngestionReport()
        seen: set[tuple[str, str]] = set()
        for item in self._source.list():
            key = (source_id, item.asset_id)
            problem = self._validate(item)
            if problem is not None:
                logger.warning("quarantined %s (%s): %s", item.uri, item.asset_id, problem)
                report = dataclasses.replace(report, quarantined=report.quarantined + 1)
                continue
            seen.add(key)
            try:
                report = self._ingest(item, held.get(key, []), now, report)
            except Exception:
                logger.exception("failed to ingest %s (%s)", item.uri, item.asset_id)
                report = dataclasses.replace(report, failed=report.failed + 1)

        for key, holders in held.items():
            if key in seen:
                continue
            for record, _ in holders:
                try:
                    self._writer.remove_source_reference(record.version, *key, at=now)
                except Exception:
                    logger.exception("failed to retire %s from %s", key[1], record.version)
                    report = dataclasses.replace(report, failed=report.failed + 1)
                    continue
                logger.info("retired %s from %s", key[1], record.version)
                report = dataclasses.replace(report, retired=report.retired + 1)

        logger.info("ingestion pass over %s: %s", source_id, report)
        return report

    def _ingest(
        self,
        item: DiscoveredMedia,
        holders: _Holders,
        now: datetime,
        report: models.IngestionReport,
    ) -> models.IngestionReport:
        if len(holders) == 1 and self._same_bytes(holders[0][1], item):
            held_record, held_reference = holders[0]
            if not self._same_metadata(held_reference, item):
                self._writer.add_source_reference(held_record.version, self._reference(item, now))
                logger.info("refreshed %s on %s", item.uri, held_record.version)
            return dataclasses.replace(report, unchanged=report.unchanged + 1)

        version = self._hash(item)
        reference = self._reference(item, now)
        same_bytes = any(record.version == version for record, _ in holders)
        if self._reader.get(version) is None:
            record = models.AssetVersion(
                version=version,
                media_type=item.media_type,
                size_bytes=item.size_bytes,
                sources=(reference,),
                first_seen_at=now,
                lineage=models.Lineage(fingerprint=self._fingerprint, version=version),
            )
            if self._writer.register_version(record):
                logger.info("registered %s as %s", item.uri, version)
            else:
                self._writer.add_source_reference(version, reference)
        else:
            self._writer.add_source_reference(version, reference)
            logger.info("added %s to %s", item.uri, version)

        superseded = 0
        for record, _ in holders:
            if record.version != version:
                self._writer.remove_source_reference(
                    record.version, reference.source_id, reference.asset_id, at=now
                )
                logger.info("superseded %s by %s for %s", record.version, version, item.asset_id)
                superseded += 1
        return dataclasses.replace(
            report,
            registered=report.registered + (0 if same_bytes else 1),
            unchanged=report.unchanged + (1 if same_bytes else 0),
            superseded=report.superseded + superseded,
        )

    def _validate(self, item: DiscoveredMedia) -> str | None:
        """Return why the item is not an acceptable asset, or ``None`` when it is."""
        if item.size_bytes < self._config.min_size_bytes:
            return f"{item.size_bytes} bytes is below min_size_bytes={self._config.min_size_bytes}"
        return None

    def _reference(self, item: DiscoveredMedia, now: datetime) -> models.SourceReference:
        return models.SourceReference(
            source_id=self._source.source_id,
            asset_id=item.asset_id,
            uri=item.uri,
            filename=item.filename,
            size_bytes=item.size_bytes,
            modified_at=item.modified_at,
            entitlements=item.entitlements,
            seen_at=now,
        )

    @staticmethod
    def _same_bytes(reference: models.SourceReference, item: DiscoveredMedia) -> bool:
        """Return whether the listing gives no reason to re-hash the item."""
        return reference.size_bytes == item.size_bytes and reference.modified_at == item.modified_at

    @staticmethod
    def _same_metadata(reference: models.SourceReference, item: DiscoveredMedia) -> bool:
        """Return whether the stored reference still describes the item as listed."""
        return (
            reference.uri == item.uri
            and reference.filename == item.filename
            and reference.entitlements == item.entitlements
        )

    def _hash(self, item: DiscoveredMedia) -> str:
        digest = hashlib.new(self._config.hash_algorithm)
        with item.open() as stream:
            while chunk := stream.read(_READ_CHUNK_BYTES):
                digest.update(chunk)
        return f"{self._config.hash_algorithm}:{digest.hexdigest()}"
