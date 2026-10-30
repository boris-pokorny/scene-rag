"""Client of the retrieval service's query-api, with its own decoder of the published responses.

Evaluation keeps no copy of retrieval's schemas: this decoder, pinned by copies of
retrieval's published samples under `tests/contract/samples/`, is the only coupling. It reads
only what a metric needs, so a backward-compatible addition to the responses changes nothing.
The caller's entitlements travel in the header query-api trusts from its gateway, so each case
is asked exactly as its caller would be.
"""

from collections.abc import Mapping
from typing import Any, Final

from scene_rag.evaluation.http_transport import Json, JsonTransport, TransportError
from scene_rag.evaluation.models import Answer, Response, Served, Source
from scene_rag.evaluation.protocols import QueryApiError

ENTITLEMENTS_HEADER: Final = "X-Scene-Rag-Entitlements"
"""Comma-separated entitlements; empty means the caller sees nothing."""


def _source(document: Mapping[str, Any]) -> Source:
    return Source(
        asset_version=str(document["asset_version"]),
        start_seconds=float(document["start_seconds"]),
        end_seconds=float(document["end_seconds"]),
    )


def _served(document: Mapping[str, Any]) -> Served:
    chunking = document["chunking_fingerprint"]
    return Served(
        query_config=str(document["query_config"]),
        embedding_fingerprint=str(document["embedding_fingerprint"]),
        chunking_fingerprint=None if chunking is None else str(chunking),
        collection=str(document["collection"]),
    )


def _optional_count(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"a token count must be an integer, not {value!r}")
    return value


def search_response_from_json(document: Json) -> Response:
    """Decode a search response.

    Raises:
        QueryApiError: When a field evaluation reads is missing or mistyped.
    """
    try:
        return Response(
            sources=tuple(_source(item) for item in document["sources"]),
            served=_served(document["served"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryApiError(f"undecodable search response: {exc!r}") from exc


def answer_response_from_json(document: Json) -> Response:
    """Decode an answer response.

    Raises:
        QueryApiError: When a field evaluation reads is missing or mistyped.
    """
    try:
        usage = document["usage"] or {}
        answer = Answer(
            text=str(document["answer"]),
            refused=bool(document["refused"]),
            citations=tuple(int(n) for n in document["citations"]),
            invalid_citations=tuple(int(n) for n in document["invalid_citations"]),
            prompt_tokens=_optional_count(usage.get("prompt_tokens")),
            completion_tokens=_optional_count(usage.get("completion_tokens")),
        )
        return Response(
            sources=tuple(_source(item) for item in document["sources"]),
            served=_served(document["served"]),
            answer=answer,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise QueryApiError(f"undecodable answer response: {exc!r}") from exc


class QueryApiClient:
    """`QueryApi` over HTTP."""

    def __init__(self, base_url: str, transport: JsonTransport, timeout_seconds: float) -> None:
        """Bind the server and how long to wait for it."""
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout_seconds

    def _post(self, path: str, payload: Json, entitlements: frozenset[str]) -> Json:
        headers = {ENTITLEMENTS_HEADER: ",".join(sorted(entitlements))}
        try:
            return self._transport.exchange(
                "POST", self._base_url + path, payload, headers, self._timeout
            )
        except TransportError as exc:
            raise QueryApiError(str(exc)) from exc

    def health(self) -> None:
        """Check that query-api answers.

        Raises:
            QueryApiError: When it does not.
        """
        try:
            self._transport.exchange("GET", self._base_url + "/health", None, {}, self._timeout)
        except TransportError as exc:
            raise QueryApiError(str(exc)) from exc

    def search(self, question: str, entitlements: frozenset[str], top_k: int) -> Response:
        """``POST /search`` under the caller's entitlements."""
        payload = {"question": question, "top_k": top_k}
        return search_response_from_json(self._post("/search", payload, entitlements))

    def answer(self, question: str, entitlements: frozenset[str]) -> Response:
        """``POST /answer`` under the caller's entitlements and the served defaults."""
        payload = {"question": question}
        return answer_response_from_json(self._post("/answer", payload, entitlements))
