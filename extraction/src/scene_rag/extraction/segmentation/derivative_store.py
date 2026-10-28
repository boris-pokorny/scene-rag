"""Derivative Store backend over the filesystem."""

import hashlib
import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scene_rag.extraction.segmentation import models

logger = logging.getLogger(__name__)

_READ_CHUNK_BYTES: Final = 1 << 20

# Media types the store files, with the extension each gets on disk.
EXTENSIONS: Final[Mapping[str, str]] = MappingProxyType({"image/jpeg": ".jpg", "audio/wav": ".wav"})


class FilesystemDerivativeStore:
    """`DerivativeStore` filing each derivative at ``<root>/<algorithm>/<hh>/<digest><ext>``.

    The two-character fan-out keeps directories small. A file is moved into place under a
    temporary name and renamed, so a reader never sees a partial file; a derivative already
    present is left as it is, since its content is its name.
    """

    def __init__(self, root: Path, algorithm: str) -> None:
        """Create a store under ``root``, hashing with ``algorithm``; the root is created."""
        self._root = root.resolve()
        self._algorithm = algorithm
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, path: Path, media_type: str) -> models.DerivativeRef:
        """File the bytes at ``path`` under their content hash; see `DerivativeStore.put`."""
        extension = EXTENSIONS.get(media_type)
        if extension is None:
            raise ValueError(f"media type {media_type!r} is not a derivative the store files")
        digest, size_bytes = self._hash(path)
        destination = self._root / self._algorithm / digest[:2] / f"{digest}{extension}"
        if destination.exists():
            path.unlink()
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            shutil.move(path, staged)
            os.replace(staged, destination)
            logger.debug("filed %s as %s", path.name, destination)
        return models.DerivativeRef(
            digest=f"{self._algorithm}:{digest}",
            uri=destination.as_uri(),
            media_type=media_type,
            size_bytes=size_bytes,
        )

    def _hash(self, path: Path) -> tuple[str, int]:
        digest = hashlib.new(self._algorithm)
        size_bytes = 0
        with path.open("rb") as stream:
            while chunk := stream.read(_READ_CHUNK_BYTES):
                digest.update(chunk)
                size_bytes += len(chunk)
        return digest.hexdigest(), size_bytes
