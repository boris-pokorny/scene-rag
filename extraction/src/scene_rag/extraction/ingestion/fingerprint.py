"""Fingerprint of the ingestion step: hyperparameters plus the code that defines its output."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Final

from scene_rag.extraction.ingestion.config import IngestionConfig

_PACKAGE_DIR = Path(__file__).parent

# The modules that decide what the step writes. Deployment code (`bootstrap`, `settings`,
# `registry`) and the interfaces (`protocols`) are left out: a changed log message or
# connection timeout must not imply reprocessing.
_FINGERPRINTED_MODULES: Final = (
    "config.py",
    "fingerprint.py",
    "models.py",
    "service.py",
    "sources.py",
)


def _canonical(config: IngestionConfig) -> dict[str, object]:
    values: dict[str, object] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        values[field.name] = dict(sorted(value.items())) if isinstance(value, Mapping) else value
    return values


def compute_fingerprint(config: IngestionConfig) -> str:
    """Hash the hyperparameters and the output-defining modules of this subpackage.

    Only this step's code is hashed, not the whole image, so changes to other steps do not
    rerun ingestion. There is no model digest because ingestion uses no model. The sources are
    read at runtime, which assumes the installed files match the repository they were built
    from.

    Args:
        config: The hyperparameters in force.

    Returns:
        The first twelve hex digits of the digest.

    Raises:
        FileNotFoundError: When a fingerprinted module was renamed without updating the list.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(_canonical(config), sort_keys=True, separators=(",", ":")).encode())
    for name in _FINGERPRINTED_MODULES:
        digest.update(name.encode())
        digest.update((_PACKAGE_DIR / name).read_bytes())
    return digest.hexdigest()[:12]
