"""Vision model backed by an Ollama server.

Ollama stands in for the model gateway of the design: it names models by tag and reports the
digest behind each tag, which is what the fingerprint pins. This module is fingerprinted, so it
holds only what decides the answer: which endpoints are called and what they are sent. The HTTP
layer lives in `ollama_transport`, outside the fingerprint.
"""

import base64
import logging
from collections.abc import Mapping, Sequence
from typing import Final

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import DescriptionConfig
from scene_rag.extraction.enrichment.ollama_transport import Json, JsonTransport, OllamaError

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS: Final = 5.0
_DEFAULT_TAG: Final = ":latest"


class OllamaClient:
    """The few Ollama endpoints the step uses: version, model list, chat."""

    def __init__(self, base_url: str, transport: JsonTransport, timeout_seconds: float) -> None:
        """Bind to the server at ``base_url``.

        Args:
            base_url: Scheme, host and port of the server, without a path.
            transport: The HTTP layer.
            timeout_seconds: Longest wait for a chat answer.
        """
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout_seconds

    def version(self) -> str:
        """Return the server version; the connectivity check at startup.

        Raises:
            OllamaError: When the server does not answer.
        """
        answer = self._transport.get(f"{self._base_url}/api/version", _PROBE_TIMEOUT_SECONDS)
        return str(answer.get("version", ""))

    def resolve_model(self, name: str) -> models.ModelRef:
        """Return the digest behind a model name, as the server holds it.

        Raises:
            OllamaError: When the server does not answer or does not hold the model.
        """
        answer = self._transport.get(f"{self._base_url}/api/tags", _PROBE_TIMEOUT_SECONDS)
        wanted = {name, name + _DEFAULT_TAG} if ":" not in name else {name}
        for entry in answer.get("models", ()):
            if {entry.get("name"), entry.get("model")} & wanted and entry.get("digest"):
                return models.ModelRef(name=name, digest=f"sha256:{entry['digest']}")
        raise OllamaError(
            f"model {name!r} is not available at {self._base_url}. Pull it with "
            f"`ollama pull {name}`."
        )

    def chat(
        self, model: str, prompt: str, images: Sequence[bytes], options: Mapping[str, object]
    ) -> str:
        """Send one user message with images and return the assistant's text.

        Raises:
            OllamaError: When the request fails or the answer has no message.
        """
        payload: Json = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image).decode() for image in images],
                }
            ],
            "options": dict(options),
            "stream": False,
        }
        answer = self._transport.post(f"{self._base_url}/api/chat", payload, self._timeout)
        message = answer.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError(f"chat answer from {self._base_url} carries no message content")
        return str(message["content"])


class OllamaVisionModel:
    """`VisionModel` over one Ollama model, sampling as the hyperparameters say."""

    def __init__(
        self, client: OllamaClient, model: models.ModelRef, config: DescriptionConfig
    ) -> None:
        """Bind to a resolved model.

        Args:
            client: The server.
            model: The model, resolved by `OllamaClient.resolve_model`.
            config: Hyperparameters; temperature, seed and token budget.
        """
        self._client = client
        self._model = model
        self._options: dict[str, object] = {
            "temperature": config.temperature,
            "seed": config.seed,
            "num_predict": config.max_tokens,
        }

    @property
    def model(self) -> models.ModelRef:
        """Return the model and its digest."""
        return self._model

    def describe(self, images: Sequence[bytes], prompt: str) -> str:
        """Answer ``prompt`` about ``images``; see `VisionModel.describe`."""
        return self._client.chat(self._model.name, prompt, images, self._options)
