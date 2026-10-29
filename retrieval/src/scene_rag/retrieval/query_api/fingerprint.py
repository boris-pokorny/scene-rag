"""The ``query_config`` id: the role's hyperparameters, its output-defining code, its model.

The query-time analogue of a step fingerprint: a change rebuilds no index but
changes the answer, so it starts a candidate release. The configuration module is not hashed
as bytes; its values are.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

from scene_rag.retrieval.query_api.config import QueryConfig

_PACKAGE_DIR = Path(__file__).parent

# The modules that decide what the role retrieves and answers; see `compute_query_config_id`.
FINGERPRINTED_MODULES: Final = (
    "models.py",
    "fingerprint.py",
    "retriever.py",
    "reranker.py",
    "refusal.py",
    "prompt.py",
    "citations.py",
    "ollama.py",
)


def compute_query_config_id(
    config: QueryConfig, modules: Sequence[str], generator_digest: str
) -> str:
    """Hash the role's hyperparameters, its output-defining modules and the generator digest.

    Only the modules deciding what is retrieved and answered are hashed, not the whole
    subpackage. Deployment code (`bootstrap`, `settings`, the clients and transports, the
    adapters, the encoders) and the interfaces (`protocols`) are never listed: a changed log
    message, error wording or timeout must not imply a new release. The served index is
    pinned by the config, so its embedding fingerprint is in the hash through the values;
    when the pin is ``None`` the id says "newest", and every answer reports which index that
    was. The sources are read at runtime, which assumes the installed files match the
    repository they were built from.

    Args:
        config: The role's hyperparameters.
        modules: File names, within this subpackage, of the modules deciding its output.
        generator_digest: Digest of the generator model's bytes, as resolved by the backend.

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
    digest.update(generator_digest.encode())
    return digest.hexdigest()[:12]
