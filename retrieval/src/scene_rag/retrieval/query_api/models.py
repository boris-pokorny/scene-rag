"""Domain model of the query-api role: callers, queries, sources, results, read views."""

import dataclasses
from datetime import datetime
from typing import Final

INDEX_SCHEMA_VERSION: Final = 1
"""The index ledger schema this role decodes; see `index_ledger`."""
INDEX_ENTRY_SCHEMA_VERSION: Final = 1
"""The point payload schema this role decodes; see `search_index`."""
SCENE_RECORD_SCHEMA_VERSION: Final = 1
"""The scene record schema this role passes through; see `scene_api_client`."""
KEYFRAME_MEDIA_TYPE: Final = "image/jpeg"

type Vector = tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class Caller:
    """Who asks: the identity the gateway asserted, reduced to what the role filters on.

    Attributes:
        entitlements: Groups the caller belongs to; a source is visible when it shares one.
            Empty sees nothing.
    """

    entitlements: frozenset[str]


ANONYMOUS: Final = Caller(frozenset())
"""The caller of a request that asserted no entitlements."""


@dataclasses.dataclass(frozen=True)
class SearchFilters:
    """What a caller may legitimately narrow a request by.

    Attributes:
        asset_version: Only sources of this asset version, when given.
    """

    asset_version: str | None = None


@dataclasses.dataclass(frozen=True)
class SearchQuery:
    """A request for the scenes relevant to a question.

    Attributes:
        question: The question, as asked.
        caller: Who asks.
        top_k: Sources wanted; the configured default when ``None``, capped by the config.
        filters: What narrows the request.
    """

    question: str
    caller: Caller
    top_k: int | None = None
    filters: SearchFilters = SearchFilters()


@dataclasses.dataclass(frozen=True)
class AnswerQuery:
    """A request for an answer written over the relevant scenes; the fields of `SearchQuery`.

    A separate class: the two use cases may grow apart (a conversation, a language).

    Attributes:
        question: The question, as asked.
        caller: Who asks.
        top_k: Sources the answer may draw on; the configured default when ``None``.
        filters: What narrows the request.
    """

    question: str
    caller: Caller
    top_k: int | None = None
    filters: SearchFilters = SearchFilters()


@dataclasses.dataclass(frozen=True)
class ModelRef:
    """A model as the backend resolves it.

    Attributes:
        name: Name the model is known by to its backend.
        digest: Digest of the model's bytes.
    """

    name: str
    digest: str


@dataclasses.dataclass(frozen=True)
class SourceRef:
    """A source system's reference to an asset version, as a chunk carries it.

    Attributes:
        source_id: Identifier of the connector that found the item.
        asset_id: Stable identifier of the item within its source.
        filename: Display name of the item in its source.
    """

    source_id: str
    asset_id: str
    filename: str


@dataclasses.dataclass(frozen=True)
class IndexView:
    """An index ledger record, as far as the role needs it: which index, embedded how.

    Attributes:
        fingerprint: Embedding fingerprint: the identifier of the index.
        collection: Collection of the Search Index holding the points.
        model: The embedding model and its digest; a query is embedded with the same.
        dimensions: Length of every vector.
        distance: Similarity the collection is built for.
        chunking_fingerprint: Chunking fingerprint the index accepted; ``None`` when any.
        query_prefix: Put before a question before it is embedded.
        truncate: Whether inputs beyond the context are cut.
        num_ctx: Context the embedding model runs with, in tokens.
        created_at: When the index was first prepared, in UTC.
    """

    fingerprint: str
    collection: str
    model: ModelRef
    dimensions: int
    distance: str
    chunking_fingerprint: str | None
    query_prefix: str
    truncate: bool
    num_ctx: int
    created_at: datetime


@dataclasses.dataclass(frozen=True)
class IndexEntryView:
    """A point's payload, as far as the role needs it to filter, rank and cite.

    Attributes:
        chunk_version: Data version of the chunk: the join key to the Chunk Store.
        scene_set_version: Data version of the scene set.
        asset_version: Content hash of the asset.
        chunking_fingerprint: Fingerprint of the chunking run that wrote the chunk.
        scene_indices: Scenes the chunk covers, ascending.
        start_seconds: Start of the covered span in the stream.
        end_seconds: End of the covered span in the stream.
        text: The retrieval text.
        sources: Source references holding the asset version, for citations.
        entitlements: Who may see the chunk.
    """

    chunk_version: str
    scene_set_version: str
    asset_version: str
    chunking_fingerprint: str
    scene_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    text: str
    sources: tuple[SourceRef, ...]
    entitlements: frozenset[str]


@dataclasses.dataclass(frozen=True)
class Hit:
    """One point the Search Index ranked.

    Attributes:
        entry: The point's payload.
        score: Similarity to the query, as the backend scores it; higher is closer.
    """

    entry: IndexEntryView
    score: float


@dataclasses.dataclass(frozen=True)
class Source:
    """A retrieved scene span, ranked and citable.

    Attributes:
        chunk_version: Data version of the chunk behind it.
        asset_version: Content hash of the asset.
        scene_set_version: Data version of the scene set, for ``scene://`` resources.
        scene_indices: Scenes the span covers, ascending.
        start_seconds: Start of the span in the stream.
        end_seconds: End of the span in the stream.
        score: Similarity to the question; higher is closer.
        text: The text the span was found by.
        sources: Source references holding the asset version.
        entitlements: Who may see the span.
    """

    chunk_version: str
    asset_version: str
    scene_set_version: str
    scene_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    score: float
    text: str
    sources: tuple[SourceRef, ...]
    entitlements: frozenset[str]


@dataclasses.dataclass(frozen=True)
class Served:
    """What produced an answer: reported with every response.

    Attributes:
        query_config: Id of the role's hyperparameters, code and generator in force.
        embedding_fingerprint: Fingerprint of the index served.
        chunking_fingerprint: Chunking fingerprint that index accepted; ``None`` when any.
        collection: Collection of the Search Index served.
    """

    query_config: str
    embedding_fingerprint: str
    chunking_fingerprint: str | None
    collection: str


@dataclasses.dataclass(frozen=True)
class SearchResult:
    """The answer to a `SearchQuery`.

    Attributes:
        sources: Relevant spans, best first.
        served: What produced them.
    """

    sources: tuple[Source, ...]
    served: Served


@dataclasses.dataclass(frozen=True)
class Message:
    """One turn of a chat prompt.

    Attributes:
        role: ``system``, ``user`` or ``assistant``.
        content: The text.
    """

    role: str
    content: str


@dataclasses.dataclass(frozen=True)
class Usage:
    """Tokens one generation consumed, when the backend reports them.

    Attributes:
        prompt_tokens: Tokens of the prompt.
        completion_tokens: Tokens of the answer.
    """

    prompt_tokens: int | None
    completion_tokens: int | None


NO_USAGE: Final = Usage(None, None)
"""The usage of an answer no model was called for."""


@dataclasses.dataclass(frozen=True)
class Generation:
    """What the generator produced.

    Attributes:
        text: The answer, as generated.
        usage: Tokens consumed.
    """

    text: str
    usage: Usage


@dataclasses.dataclass(frozen=True)
class AnswerResult:
    """The answer to an `AnswerQuery`.

    Attributes:
        answer: The answer, or the refusal.
        refused: Whether the role declined to answer.
        citations: Source numbers the answer cites that name a source, one-based, in order
            of first mention.
        invalid_citations: Source numbers the answer cites that name no source.
        sources: The spans the answer could draw on, best first; ``[n]`` is ``sources[n-1]``.
        usage: Tokens consumed; none for a refusal.
        served: What produced the answer.
    """

    answer: str
    refused: bool
    citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    sources: tuple[Source, ...]
    usage: Usage
    served: Served


@dataclasses.dataclass(frozen=True)
class DerivativeContent:
    """The bytes of a keyframe, with their media type.

    Attributes:
        media_type: What the bytes are.
        data: The bytes.
    """

    media_type: str
    data: bytes


@dataclasses.dataclass(frozen=True)
class CollectionInfo:
    """What a Search Index backend reports about a collection.

    Attributes:
        dimensions: Length of the vectors.
        distance: Similarity, in the index's names.
        point_count: Points stored.
    """

    dimensions: int
    distance: str
    point_count: int
