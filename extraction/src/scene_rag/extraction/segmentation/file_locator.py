"""`MediaLocator` over the local filesystem: where an asset version's bytes can be read from."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.models import truncate_to_millisecond
from scene_rag.extraction.segmentation.protocols import MediaUnavailableError

logger = logging.getLogger(__name__)


def to_millisecond_utc(timestamp: float) -> datetime:
    """Convert a POSIX timestamp to a UTC datetime truncated to milliseconds, as ingestion does."""
    return truncate_to_millisecond(datetime.fromtimestamp(timestamp, UTC))


class FileUriLocator:
    """`MediaLocator` over ``file://`` source references.

    A file is accepted only when its size and modification time still match what the
    connector recorded when it hashed the bytes: a file rewritten since then no longer holds
    the version, and ingestion will supersede the reference on its next pass.
    """

    def locate(self, record: models.AssetVersionView) -> Path:
        """Return the first source file that still looks like the version; see the protocol."""
        reasons: list[str] = []
        for source in record.sources:
            parsed = urlparse(source.uri)
            if parsed.scheme != "file":
                reasons.append(f"{source.uri}: not a file URI")
                continue
            path = Path(unquote(parsed.path))
            try:
                status = path.stat()
            except OSError as exc:
                reasons.append(f"{source.uri}: {exc}")
                continue
            if status.st_size != source.size_bytes:
                reasons.append(f"{source.uri}: size changed since ingestion")
                continue
            if to_millisecond_utc(status.st_mtime) != source.modified_at:
                reasons.append(f"{source.uri}: modified since ingestion")
                continue
            return path
        raise MediaUnavailableError(
            f"no source of {record.version} is readable as recorded: " + "; ".join(reasons)
        )
