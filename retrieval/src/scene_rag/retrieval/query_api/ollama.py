"""Query embedder and answer generator backed by an Ollama server.

Ollama stands in for a model gateway: it names models by tag and reports the
digest behind each tag, which the index record and the ``query_config`` id pin. This module
is part of the ``query_config`` id, so it holds only what decides a vector or an answer:
which endpoint is called and what it is sent. The HTTP layer lives in `http_transport`,
pulling a model in `ollama_pull`, both outside the id.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Final

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.config import GeneratorConfig
from scene_rag.retrieval.query_api.http_transport import Json, JsonTransport, TransportError
from scene_rag.retrieval.query_api.protocols import ModelError

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS: Final = 5.0
_DEFAULT_TAG: Final = ":latest"


class OllamaClient:
    """The few Ollama endpoints the role uses: version, model list, embed, chat."""

    def __init__(
        self,
        base_url: str,
        transport: JsonTransport,
        *,
        embed_timeout_seconds: float,
        chat_timeout_seconds: float,
    ) -> None:
        """Bind to the server at ``base_url``.

        Args:
            base_url: Scheme, host and port of the server, without a path.
            transport: The HTTP layer.
            embed_timeout_seconds: Longest wait for an embedding answer.
            chat_timeout_seconds: Longest wait for a generated answer.
        """
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._embed_timeout = embed_timeout_seconds
        self._chat_timeout = chat_timeout_seconds

    @property
    def base_url(self) -> str:
        """Return the server's base URL."""
        return self._base_url

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
        answer = self._exchange("POST", "/api/embed", payload, self._embed_timeout)
        embeddings = answer.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise ModelError(
                f"embed answer from {self._base_url} carries "
                f"{'no' if not isinstance(embeddings, list) else len(embeddings)} vector(s) "
                f"for {len(inputs)} input(s)"
            )
        return tuple(tuple(float(value) for value in vector) for vector in embeddings)

    def chat(
        self, model: str, messages: Sequence[models.Message], options: Mapping[str, object]
    ) -> models.Generation:
        """Answer a chat prompt in one request, without streaming.

        Raises:
            ModelError: When the request fails or the answer carries no message content.
        """
        payload: Json = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": dict(options),
            "stream": False,
        }
        answer = self._exchange("POST", "/api/chat", payload, self._chat_timeout)
        message = answer.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ModelError(f"chat answer from {self._base_url} carries no message content")
        return models.Generation(
            text=str(message["content"]),
            usage=models.Usage(
                prompt_tokens=_count(answer.get("prompt_eval_count")),
                completion_tokens=_count(answer.get("eval_count")),
            ),
        )


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class OllamaQueryEmbedder:
    """`QueryEmbedder` embedding questions exactly as the served index was built."""

    def __init__(self, client: OllamaClient, index: models.IndexView) -> None:
        """Bind to the index's model, prefix, truncation and context.

        Args:
            client: The server.
            index: The index served; its model digest was verified against the server by
                `bootstrap`.
        """
        self._client = client
        self._index = index

    @property
    def model(self) -> models.ModelRef:
        """Return the embedding model and its digest."""
        return self._index.model

    @property
    def dimensions(self) -> int:
        """Return the length of the vectors."""
        return self._index.dimensions

    def embed(self, question: str) -> models.Vector:
        """Return the question's vector behind the index's query prefix."""
        (vector,) = self._client.embed(
            self._index.model.name,
            [self._index.query_prefix + question],
            truncate=self._index.truncate,
            num_ctx=self._index.num_ctx,
        )
        if len(vector) != self._index.dimensions:
            raise ModelError(
                f"{self._index.model.name} answered {len(vector)} dimensions, the index "
                f"{self._index.collection} holds {self._index.dimensions}"
            )
        return vector


class OllamaAnswerGenerator:
    """`AnswerGenerator` over one Ollama chat model, sampling as the config says."""

    def __init__(
        self, client: OllamaClient, model: models.ModelRef, config: GeneratorConfig
    ) -> None:
        """Bind to a resolved model.

        Args:
            client: The server.
            model: The model, resolved by `OllamaClient.resolve_model`.
            config: Hyperparameters in force.
        """
        self._client = client
        self._model = model
        self._config = config

    @property
    def model(self) -> models.ModelRef:
        """Return the generator model and its digest."""
        return self._model

    def generate(self, messages: Sequence[models.Message]) -> models.Generation:
        """Return the answer to the prompt; see `AnswerGenerator.generate`."""
        options = {
            "temperature": self._config.temperature,
            "seed": self._config.seed,
            "num_predict": self._config.max_tokens,
            "num_ctx": self._config.num_ctx,
        }
        generation = self._client.chat(self._model.name, messages, options)
        logger.debug(
            "generated %d character(s), %s prompt and %s completion tokens",
            len(generation.text),
            generation.usage.prompt_tokens,
            generation.usage.completion_tokens,
        )
        return generation
