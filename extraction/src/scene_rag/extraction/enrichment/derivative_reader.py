"""`DerivativeReader` over the local filesystem, verifying every file against its digest."""

import hashlib
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.protocols import DerivativeUnavailableError

_READ_CHUNK_BYTES: Final = 1 << 20


class FileDerivativeReader:
    """Reads ``file://`` derivatives and refuses any whose bytes no longer match their digest.

    The Derivative Store is content-addressed, so a mismatch means the file was replaced or
    corrupted; enriching it would attribute text to bytes nobody filed.
    """

    def read(self, reference: models.DerivativeRefView) -> bytes:
        """Return the verified bytes; see `DerivativeReader.read`."""
        parsed = urlparse(reference.uri)
        if parsed.scheme != "file":
            raise DerivativeUnavailableError(f"{reference.uri}: not a file URI")
        algorithm, _, expected = reference.digest.partition(":")
        if not expected or algorithm not in hashlib.algorithms_available:
            raise DerivativeUnavailableError(
                f"{reference.uri}: unusable digest {reference.digest!r}"
            )
        try:
            data = Path(unquote(parsed.path)).read_bytes()
        except OSError as exc:
            raise DerivativeUnavailableError(f"{reference.uri}: {exc}") from exc
        if hashlib.new(algorithm, data).hexdigest() != expected:
            raise DerivativeUnavailableError(f"{reference.uri}: content does not match its digest")
        return data
