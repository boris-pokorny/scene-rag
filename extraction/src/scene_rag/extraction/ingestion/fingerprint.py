"""Fingerprint of the ingestion step: hyperparameters plus the step's code."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

from scene_rag.extraction.ingestion.config import IngestionConfig

_PACKAGE_DIR = Path(__file__).parent


def _canonical(config: IngestionConfig) -> dict[str, object]:
    values: dict[str, object] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        values[field.name] = dict(sorted(value.items())) if isinstance(value, Mapping) else value
    return values


def compute_fingerprint(config: IngestionConfig) -> str:
    """Hash the hyperparameters and every module of this subpackage.

    Only this step's code is hashed, not the whole image, so changes to other steps do not
    rerun ingestion. There is no model digest because ingestion uses no model.

    Args:
        config: The hyperparameters in force.

    Returns:
        The first twelve hex digits of the digest.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(_canonical(config), sort_keys=True, separators=(",", ":")).encode())
    for module in sorted(_PACKAGE_DIR.glob("*.py")):
        digest.update(module.name.encode())
        digest.update(module.read_bytes())
    return digest.hexdigest()[:12]
