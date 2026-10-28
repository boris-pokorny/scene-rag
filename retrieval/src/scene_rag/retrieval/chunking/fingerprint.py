"""Fingerprint of the chunking step: its hyperparameters and the code defining its output.

The step has no model, so nothing else joins the hash. The accepted upstream fingerprints are
part of the hyperparameters, so a new Media model reruns chunking under a new fingerprint.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from scene_rag.retrieval.chunking.config import ChunkingConfig

_PACKAGE_DIR = Path(__file__).parent


def compute_fingerprint(config: ChunkingConfig, modules: Sequence[str]) -> str:
    """Hash the step's hyperparameters and its output-defining modules.

    Only the modules deciding what a chunk contains are hashed, not the whole subpackage.
    Deployment code (`bootstrap`, `settings`, `chunk_store`, `scene_api_client`,
    `http_transport`), orchestration (`scheduler`, `service`, `dispatch`, `report`) and the
    interfaces (`protocols`) are never listed: a changed log message, error wording, pass
    report, timeout or worker count must not imply reprocessing. `models` carries the request
    and liveness views (`SceneSetQuery`, `AssetVersionView`) alongside the output types; they
    change with the published API, rarely, and are accepted in the hash. The sources are read
    at runtime, which assumes the installed files match the repository they were built from.

    Args:
        config: The step's hyperparameters.
        modules: File names, within this subpackage, of the modules deciding its output.

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
    return digest.hexdigest()[:12]
