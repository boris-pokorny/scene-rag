"""Fingerprint of one enricher: its hyperparameters, the code defining its output, its model.

Each enricher is fingerprinted on its own, so a changed prompt reruns the descriptions and
nothing else. The configuration module is not hashed as bytes, since every enricher's section
lives in it; the values of the enricher's own section are hashed instead.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from scene_rag.extraction.enrichment.config import DescriptionConfig, TranscriptionConfig

_PACKAGE_DIR = Path(__file__).parent

type Section = DescriptionConfig | TranscriptionConfig
"""A section of `EnrichmentConfig`: the hyperparameters of one enricher."""


def compute_fingerprint(section: Section, modules: Sequence[str], model_digest: str) -> str:
    """Hash an enricher's hyperparameters, its output-defining modules and its model digest.

    Only the enricher's own modules are hashed, not the whole image or subpackage, so a change
    to one enricher does not rerun its siblings. Deployment code (`bootstrap`, `settings`, the
    store backends, `ollama_transport`, `whisper_loader`), orchestration (`scheduler`,
    `service`, `dispatch`) and the interfaces (`protocols`) are never listed: a changed log
    message, error wording, connection timeout or worker count must not imply reprocessing.
    The sources are read at runtime, which assumes the installed files match the repository
    they were built from.

    Args:
        section: The enricher's hyperparameters.
        modules: File names, within this subpackage, of the modules deciding its output.
        model_digest: Digest of the model's bytes, as resolved by the backend.

    Returns:
        The first twelve hex digits of the digest.

    Raises:
        FileNotFoundError: When a listed module was renamed without updating the list.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(section), sort_keys=True, separators=(",", ":")).encode())
    for name in modules:
        digest.update(name.encode())
        digest.update((_PACKAGE_DIR / name).read_bytes())
    digest.update(model_digest.encode())
    return digest.hexdigest()[:12]
