"""Embedding model backed by an Ollama server.

Ollama stands in for a model gateway: it names models by tag and reports the
digest behind each tag, which is what the fingerprint pins. This module is fingerprinted, so it
holds only what decides the vector: which endpoint is called, what it is sent, and what is put
before the text. The HTTP layer lives in `http_transport`, outside the fingerprint.
"""

import logging
from collections.abc import Sequence
from typing import Final

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.config import EmbeddingConfig
from scene_rag.retrieval.embedding.fingerprint import compute_fingerprint
from scene_rag.retrieval.embedding.http_transport import Json, JsonTransport, TransportError
from scene_rag.retrieval.embedding.protocols import ModelError

logger = logging.getLogger(__name__)

NAME: Final = "ollama"
# The modules that decide what this model writes; see `fingerprint.compute_fingerprint`.
FINGERPRINTED_MODULES: Final = ("models.py", "fingerprint.py", "ollama_embedding.py")

_PROBE_TIMEOUT_SECONDS: Final = 5.0
_DEFAULT_TAG: Final = ":latest"
_DIMENSION_PROBE: Final = "dimension probe"


class OllamaClient:
    """The few Ollama endpoints the step uses: version, model list, embed."""

    def __init__(self, base_url: str, transport: JsonTransport, timeout_seconds: float) -> None:
        """Bind to the server at ``base_url``.

        Args:
            base_url: Scheme, host and port of the server, without a path.
            transport: The HTTP layer.
            timeout_seconds: Longest wait for an embedding answer.
        """
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout_seconds

    def _exchange(self, method: str, path: str, payload: Json | None, timeout: float) -> Json:
        try:
            return self._transport.exchange(method, self._base_url + path, payload, timeout)
        except TransportError as exc:
            raise ModelError(str(exc)) from exc

    def version(self) -> str:
        """Return the server version; the connectivity check at startup.

        Raises:
            ModelError: When the server does not answer.
        """
        answer = self._exchange("GET", "/api/version", None, _PROBE_TIMEOUT_SECONDS)
        return str(answer.get("version", ""))

    def resolve_model(self, name: str) -> models.ModelRef:
        """Return the digest behind a model name, as the server holds it.

        Raises:
            ModelError: When the server does not answer or does not hold the model.
        """
        answer = self._exchange("GET", "/api/tags", None, _PROBE_TIMEOUT_SECONDS)
        wanted = {name, name + _DEFAULT_TAG} if ":" not in name else {name}
        for entry in answer.get("models", ()):
            if {entry.get("name"), entry.get("model")} & wanted and entry.get("digest"):
                return models.ModelRef(name=name, digest=f"sha256:{entry['digest']}")
        raise ModelError(
            f"model {name!r} is not available at {self._base_url}. Pull it with "
            f"`ollama pull {name}`."
        )

    def embed(
        self, model: str, inputs: Sequence[str], *, truncate: bool, num_ctx: int
    ) -> tuple[models.Vector, ...]:
        """Embed the inputs in one request and return one vector per input, in order.

        Raises:
            ModelError: When the request fails or the answer has the wrong number of vectors.
        """
        payload: Json = {
            "model": model,
            "input": list(inputs),
            "truncate": truncate,
            "options": {"num_ctx": num_ctx},
        }
        answer = self._exchange("POST", "/api/embed", payload, self._timeout)
        embeddings = answer.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise ModelError(
                f"embed answer from {self._base_url} carries "
                f"{'no' if not isinstance(embeddings, list) else len(embeddings)} vector(s) "
                f"for {len(inputs)} input(s)"
            )
        logger.debug(
            "embedded %d input(s), %s tokens", len(inputs), answer.get("prompt_eval_count")
        )
        return tuple(tuple(float(value) for value in vector) for vector in embeddings)


class OllamaEmbeddingModel:
    """`EmbeddingModel` over one Ollama model, prefixing and truncating as the config says."""

    def __init__(
        self, client: OllamaClient, model: models.ModelRef, config: EmbeddingConfig
    ) -> None:
        """Bind to a resolved model and compute the step's fingerprint.

        Args:
            client: The server.
            model: The model, resolved by `OllamaClient.resolve_model`.
            config: Hyperparameters in force.
        """
        self._client = client
        self._model = model
        self._config = config.model
        self._fingerprint = compute_fingerprint(config, FINGERPRINTED_MODULES, model.digest)
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        """Return ``"ollama"``."""
        return NAME

    @property
    def model(self) -> models.ModelRef:
        """Return the model and its digest."""
        return self._model

    @property
    def fingerprint(self) -> str:
        """Return the step's fingerprint."""
        return self._fingerprint

    @property
    def dimensions(self) -> int:
        """Return the vector length, asked of the server once with a probe text."""
        if self._dimensions is None:
            (vector,) = self.embed((_DIMENSION_PROBE,))
            self._dimensions = len(vector)
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> tuple[models.Vector, ...]:
        """Embed the texts behind the document prefix; see `EmbeddingModel.embed`."""
        if not texts:
            return ()
        return self._client.embed(
            self._model.name,
            [self._config.document_prefix + text for text in texts],
            truncate=self._config.truncate,
            num_ctx=self._config.num_ctx,
        )
