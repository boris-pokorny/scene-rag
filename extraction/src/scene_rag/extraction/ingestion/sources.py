"""Connectors: `MediaSource` implementations over source systems."""

import dataclasses
import hashlib
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from scene_rag.extraction.ingestion.config import IngestionConfig

logger = logging.getLogger(__name__)


def asset_id_for(source_id: str, item_path: str) -> str:
    """Derive the stable asset id of an item from its source and its path within it.

    Args:
        source_id: Identifier of the connector.
        item_path: Path of the item relative to the source root, with ``/`` separators.

    Returns:
        ``asset-`` followed by sixteen hex digits.
    """
    digest = hashlib.sha256(f"{source_id}\n{item_path}".encode()).hexdigest()
    return f"asset-{digest[:16]}"


def to_millisecond_utc(timestamp: float) -> datetime:
    """Convert a POSIX timestamp to a UTC datetime truncated to milliseconds.

    The registry stores BSON dates, which hold milliseconds; truncating before comparing
    keeps a round-tripped ``modified_at`` equal to a fresh listing's.
    """
    moment = datetime.fromtimestamp(timestamp, UTC)
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


@dataclasses.dataclass(frozen=True)
class FilesystemMedia:
    """A file found under the connector's root; satisfies `DiscoveredMedia`.

    Attributes:
        asset_id: Stable id derived from the source id and the relative path.
        uri: ``file://`` URI of the resolved path.
        filename: Base name of the file.
        media_type: MIME type from the extension, per `IngestionConfig.media_types`.
        size_bytes: Size at listing time.
        modified_at: Modification time at listing time, UTC, millisecond precision.
        entitlements: The connector's static entitlements.
        path: Where to read the bytes.
    """

    asset_id: str
    uri: str
    filename: str
    media_type: str
    size_bytes: int
    modified_at: datetime
    entitlements: frozenset[str]
    path: Path

    def open(self) -> BinaryIO:
        """Open the file for binary reading."""
        return self.path.open("rb")


class FilesystemMediaSource:
    """`MediaSource` listing video files under a directory.

    The filesystem has no permission model, so every item carries the entitlements the
    connector was configured with.
    """

    def __init__(
        self,
        root: Path,
        source_id: str,
        entitlements: frozenset[str],
        config: IngestionConfig,
    ) -> None:
        """Create a connector over ``root``.

        Args:
            root: Directory to list; must exist.
            source_id: Identifier stamped on every reference.
            entitlements: Granted to every listed item.
            config: Supplies the accepted media types and the recursion flag.
        """
        self._root = root.resolve()
        self._source_id = source_id
        self._entitlements = entitlements
        self._media_types = config.media_types
        self._recursive = config.recursive

    @property
    def source_id(self) -> str:
        """Identifier stamped on every reference."""
        return self._source_id

    def list(self) -> Iterator[FilesystemMedia]:
        """Yield every accepted file under the root, in path order.

        A file that disappears between the directory walk and its ``stat`` is logged and
        skipped; the next pass sees the directory as it is then.

        Raises:
            FileNotFoundError: When the root itself is gone (unmounted, say). An empty
                listing would retire every asset of the source, so the pass must not run.
        """
        if not self._root.is_dir():
            raise FileNotFoundError(f"source root is not a directory: {self._root}")
        logger.info("listing %s for media files", self._root)
        pattern = "**/*" if self._recursive else "*"
        for path in sorted(self._root.glob(pattern)):
            media_type = self._media_types.get(path.suffix.lower())
            if media_type is None:
                continue
            try:
                if not path.is_file():
                    continue
                status = path.stat()
            except OSError as exc:
                logger.warning("skipped %s: %s", path, exc)
                continue
            relative = path.relative_to(self._root).as_posix()
            yield FilesystemMedia(
                asset_id=asset_id_for(self._source_id, relative),
                uri=path.as_uri(),
                filename=path.name,
                media_type=media_type,
                size_bytes=status.st_size,
                modified_at=to_millisecond_utc(status.st_mtime),
                entitlements=self._entitlements,
                path=path,
            )
