"""Pulling the generator model into the Ollama server: the ``pull-models`` role's action here.

Deployment code, deliberately outside the ``query_config`` id, like `http_transport`: how a
model reaches the server changes nothing about what it answers. The model name comes from
`GeneratorConfig`, so a deployment never names it a second time.
"""

import logging
from typing import Final

from scene_rag.retrieval.query_api.http_transport import JsonTransport, TransportError
from scene_rag.retrieval.query_api.protocols import ModelError

logger = logging.getLogger(__name__)

# A pull answers only once the whole model is on disk; a stalled download should still end
# the process eventually.
PULL_TIMEOUT_SECONDS: Final = 3600.0


def pull_model(base_url: str, name: str, transport: JsonTransport) -> None:
    """Ask the server at ``base_url`` to download the model ``name`` from its registry.

    Args:
        base_url: Scheme, host and port of the server, without a path.
        name: The model, as the server names it (``qwen2.5:3b``).
        transport: The HTTP layer.

    Raises:
        ModelError: When the server does not answer or does not report success.
    """
    url = f"{base_url.rstrip('/')}/api/pull"
    try:
        answer = transport.exchange(
            "POST", url, {"model": name, "stream": False}, PULL_TIMEOUT_SECONDS
        )
    except TransportError as exc:
        raise ModelError(str(exc)) from exc
    status = answer.get("status")
    if status != "success":
        raise ModelError(f"pulling {name!r} at {url} ended with status {status!r}")
    logger.info("pulled %s", name)
