"""JSON encoding of what the role publishes: the shapes of ``contracts/published``.

The only module naming the published field names; the contract tests pin it to the schemas.
The shapes are typed dictionaries so the MCP adapter can declare them as structured output:
HTTP and MCP answers are then the same documents by construction.
"""

from collections.abc import Mapping
from typing import Final, TypedDict

from scene_rag.retrieval.query_api import models

TITLE: Final = "scene-rag retrieval: query-api"
API_VERSION: Final = "1.0.0"

BACKEND_MESSAGES: Final[Mapping[str, str]] = {
    "index_unavailable": "the search index cannot be read",
    "model_unavailable": "the model backend cannot answer",
    "scene_api_unavailable": "the scene-api cannot answer",
}
"""What a caller is told when a backend fails, by error code. The failure itself names
internal URLs and backend answers, so it goes to the log only."""


class SourceRefDocument(TypedDict):
    """A source system's reference to the asset version."""

    source_id: str
    asset_id: str
    filename: str


class SourceDocument(TypedDict):
    """A retrieved scene span, ranked and citable.

    The span's entitlements are not published: they would name to the caller groups the
    caller does not belong to.
    """

    chunk_version: str
    asset_version: str
    scene_set_version: str
    scene_indices: list[int]
    start_seconds: float
    end_seconds: float
    score: float
    text: str
    sources: list[SourceRefDocument]


class ServedDocument(TypedDict):
    """What produced the answer: the query config and the index."""

    query_config: str
    embedding_fingerprint: str
    chunking_fingerprint: str | None
    collection: str


class SearchResponseDocument(TypedDict):
    """The answer to a search: sources best first."""

    sources: list[SourceDocument]
    served: ServedDocument


class UsageDocument(TypedDict):
    """Tokens one generation consumed, when known."""

    prompt_tokens: int | None
    completion_tokens: int | None


class AnswerResponseDocument(TypedDict):
    """The answer to a question: text, citations into ``sources``, or a refusal."""

    answer: str
    refused: bool
    citations: list[int]
    invalid_citations: list[int]
    sources: list[SourceDocument]
    usage: UsageDocument
    served: ServedDocument


class HealthDocument(TypedDict):
    """Liveness, with what the process serves."""

    status: str
    served: ServedDocument


class ErrorBody(TypedDict):
    """Code and message of a failure."""

    code: str
    message: str


class ErrorDocument(TypedDict):
    """The body of every non-2xx answer."""

    error: ErrorBody


def encode_source(source: models.Source) -> SourceDocument:
    """Encode a source."""
    return {
        "chunk_version": source.chunk_version,
        "asset_version": source.asset_version,
        "scene_set_version": source.scene_set_version,
        "scene_indices": list(source.scene_indices),
        "start_seconds": source.start_seconds,
        "end_seconds": source.end_seconds,
        "score": source.score,
        "text": source.text,
        "sources": [
            {"source_id": s.source_id, "asset_id": s.asset_id, "filename": s.filename}
            for s in source.sources
        ],
    }


def encode_served(served: models.Served) -> ServedDocument:
    """Encode what is served."""
    return {
        "query_config": served.query_config,
        "embedding_fingerprint": served.embedding_fingerprint,
        "chunking_fingerprint": served.chunking_fingerprint,
        "collection": served.collection,
    }


def encode_search_result(result: models.SearchResult) -> SearchResponseDocument:
    """Encode a search result."""
    return {
        "sources": [encode_source(s) for s in result.sources],
        "served": encode_served(result.served),
    }


def encode_answer_result(result: models.AnswerResult) -> AnswerResponseDocument:
    """Encode an answer result."""
    return {
        "answer": result.answer,
        "refused": result.refused,
        "citations": list(result.citations),
        "invalid_citations": list(result.invalid_citations),
        "sources": [encode_source(s) for s in result.sources],
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
        },
        "served": encode_served(result.served),
    }


def encode_health(served: models.Served) -> HealthDocument:
    """Encode the health answer."""
    return {"status": "ok", "served": encode_served(served)}


def encode_error(code: str, message: str) -> ErrorDocument:
    """Encode an error body."""
    return {"error": {"code": code, "message": message}}
