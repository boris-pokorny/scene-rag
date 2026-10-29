"""Fingerprint of the embedding step: its hyperparameters, the code defining its output, its model.

The accepted chunking fingerprint is part of the hyperparameters, so a new chunker reruns
embedding under a new fingerprint. The configuration module is not hashed as bytes; its
values are.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from scene_rag.retrieval.embedding.config import EmbeddingConfig

_PACKAGE_DIR = Path(__file__).parent


def compute_fingerprint(config: EmbeddingConfig, modules: Sequence[str], model_digest: str) -> str:
    """Hash the step's hyperparameters, its output-defining modules and its model digest.

    Only the modules deciding what a vector and its entry contain are hashed, not the whole
    subpackage. Deployment code (`bootstrap`, `settings`, `chunk_reader`, `index_store`,
    `qdrant_index`, `http_transport`, `ollama_pull`, `index_setup`), orchestration
    (`scheduler`, `service`, `dispatch`, `report`) and the interfaces (`protocols`) are never
    listed: a changed log message, error wording, timeout, batch size or worker count must
    not imply re-embedding. The sources are read at runtime, which assumes the installed
    files match the repository they were built from.

    Args:
        config: The step's hyperparameters.
        modules: File names, within this subpackage, of the modules deciding its output.
        model_digest: Digest of the model's bytes, as resolved by the backend.

    Returns:
        The first twelve hex digits of the digest.

    Raises:
        FileNotFoundError: When a listed module was renamed without updating the list.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode())
    for name in modules:
        digest.update(name.encode())
        digest.update((_PACKAGE_DIR / name).read_bytes())
    digest.update(model_digest.encode())
    return digest.hexdigest()[:12]
