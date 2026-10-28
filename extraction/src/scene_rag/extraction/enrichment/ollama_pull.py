"""Pulling the vision model into the Ollama server: the ``pull-models`` role's one action.

Deployment code, deliberately outside the description fingerprint, like `ollama_transport`:
how a model reaches the server changes nothing about what it answers. The model name comes
from `EnrichmentConfig`, so a deployment never names it a second time.
"""

import logging
from typing import Final

from scene_rag.extraction.enrichment.ollama_transport import JsonTransport, OllamaError

logger = logging.getLogger(__name__)

# A pull answers only once the whole model is on disk; a few gigabytes over a slow link take
# a while, and a stalled download should still end the process eventually.
PULL_TIMEOUT_SECONDS: Final = 3600.0


def pull_model(base_url: str, name: str, transport: JsonTransport) -> None:
    """Ask the server at ``base_url`` to download the model ``name`` from its registry.

    Args:
        base_url: Scheme, host and port of the server, without a path.
        name: The model, as the server names it (``qwen2.5vl:3b``).
        transport: The HTTP layer.

    Raises:
        OllamaError: When the server does not answer or does not report success.
    """
    url = f"{base_url.rstrip('/')}/api/pull"
    answer = transport.post(url, {"model": name, "stream": False}, PULL_TIMEOUT_SECONDS)
    status = answer.get("status")
    if status != "success":
        raise OllamaError(f"pulling {name!r} at {url} ended with status {status!r}")
    logger.info("pulled %s", name)
