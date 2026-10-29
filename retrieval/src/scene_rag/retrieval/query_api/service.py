"""The use cases of the query-api role: search, answer, and the two MCP resources.

`QueryServiceImpl` holds the rules every adapter shares: what a request may ask, that a
caller without entitlements sees nothing (and costs no model call), that retired assets are
dropped, when an answer is refused, and that citations are checked against the sources.
Orchestration only: each decision is a collaborator behind a protocol, composed by
`bootstrap`.
"""

import logging

from scene_rag.retrieval.query_api import citations, models
from scene_rag.retrieval.query_api.config import QueryConfig
from scene_rag.retrieval.query_api.prompt import PromptBuilder
from scene_rag.retrieval.query_api.protocols import (
    AnswerGenerator,
    InvalidQueryError,
    Json,
    NotFoundError,
    RefusalPolicy,
    Reranker,
    RetiredFilter,
    Retriever,
    SceneApiError,
    SceneReader,
)

logger = logging.getLogger(__name__)


class QueryServiceImpl:
    """`QueryService` over the injected policies and read models."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker,
        refusal: RefusalPolicy,
        prompt: PromptBuilder,
        generator: AnswerGenerator,
        scenes: SceneReader,
        retired: RetiredFilter,
        config: QueryConfig,
        served: models.Served,
    ) -> None:
        """Compose the use cases.

        Args:
            retriever: Finds the relevant spans.
            reranker: Reorders them for the answer.
            refusal: Decides whether to answer at all.
            prompt: Renders the generator's prompt.
            generator: Writes the answer.
            scenes: Read model over the scene-api, for the resources.
            retired: Blocks asset versions retired since indexing.
            config: Hyperparameters in force.
            served: What this process serves, reported with every result.
        """
        self._retriever = retriever
        self._reranker = reranker
        self._refusal = refusal
        self._prompt = prompt
        self._generator = generator
        self._scenes = scenes
        self._retired = retired
        self._config = config
        self._served = served

    @property
    def served(self) -> models.Served:
        """Return what this process serves."""
        return self._served

    def _check(self, question: str, top_k: int | None) -> int:
        """Validate the request and return the count to retrieve.

        Raises:
            InvalidQueryError: When the question is blank or too long, or the count below one.
        """
        retriever = self._config.retriever
        if not question.strip():
            raise InvalidQueryError("question must not be blank")
        if len(question) > retriever.max_question_chars:
            raise InvalidQueryError(
                f"question must be at most {retriever.max_question_chars} characters"
            )
        if top_k is None:
            return retriever.top_k
        if top_k < 1:
            raise InvalidQueryError("top_k must be at least 1")
        return min(top_k, retriever.max_top_k)

    def _retrieve(
        self, question: str, top_k: int, caller: models.Caller, filters: models.SearchFilters
    ) -> tuple[models.Source, ...]:
        if not caller.entitlements:
            return ()
        sources = self._retriever.retrieve(
            question, top_k=top_k, caller=caller, asset_version=filters.asset_version
        )
        return tuple(s for s in sources if not self._retired.is_retired(s.asset_version))

    def search(self, query: models.SearchQuery) -> models.SearchResult:
        """Return the relevant visible spans; see `QueryService.search`."""
        top_k = self._check(query.question, query.top_k)
        sources = self._retrieve(query.question, top_k, query.caller, query.filters)
        return models.SearchResult(sources=sources, served=self._served)

    def answer(self, query: models.AnswerQuery) -> models.AnswerResult:
        """Return a cited answer, or a refusal; see `QueryService.answer`."""
        top_k = self._check(query.question, query.top_k)
        retrieved = self._retrieve(query.question, top_k, query.caller, query.filters)
        sources = tuple(self._reranker.rerank(query.question, retrieved))
        if self._refusal.should_refuse(sources):
            return models.AnswerResult(
                answer=self._config.refusal.message,
                refused=True,
                citations=(),
                invalid_citations=(),
                sources=sources,
                usage=models.NO_USAGE,
                served=self._served,
            )
        generation = self._generator.generate(self._prompt.build(query.question, sources))
        cited = citations.parse_citations(generation.text)
        valid, invalid = citations.validate(cited, len(sources))
        if invalid:
            logger.warning("answer cites %s; only %d source(s) given", invalid, len(sources))
        return models.AnswerResult(
            answer=generation.text,
            refused=False,
            citations=valid,
            invalid_citations=invalid,
            sources=sources,
            usage=generation.usage,
            served=self._served,
        )

    def get_scene(self, scene_set_version: str, index: int, caller: models.Caller) -> Json:
        """Return the scene record the caller may see; see `QueryService.get_scene`."""
        key = f"{scene_set_version}/{index}"
        record = self._scenes.get_scene_record(scene_set_version, index)
        if record is None:
            raise NotFoundError("scene", key)
        try:
            entitlements = frozenset(record["asset"]["entitlements"])
            asset_version = str(record["asset_version"])
        except (KeyError, TypeError) as exc:
            raise SceneApiError(f"scene record {key} carries no asset entitlements") from exc
        if not entitlements & caller.entitlements or self._retired.is_retired(asset_version):
            # Not visible is not distinguishable from absent: no probing for restricted scenes.
            raise NotFoundError("scene", key)
        return record

    def get_keyframe(self, digest: str, caller: models.Caller) -> models.DerivativeContent:
        """Return the keyframe's bytes; see `QueryService.get_keyframe`.

        A caller without entitlements sees nothing, as everywhere else. Beyond that the
        entitlements are not matched: a keyframe digest is content-addressed and only learned
        from a scene record the caller was entitled to. A derivative of another media type is
        not a keyframe and is reported absent.
        """
        if not caller.entitlements:
            raise NotFoundError("keyframe", digest)
        content = self._scenes.get_derivative(digest)
        if content is None or content.media_type != models.KEYFRAME_MEDIA_TYPE:
            raise NotFoundError("keyframe", digest)
        return content


class NoService:
    """`QueryService` that is never called: the adapters are introspected, not run."""

    @property
    def served(self) -> models.Served:
        """Never answered."""
        raise NotImplementedError

    def search(self, query: models.SearchQuery) -> models.SearchResult:
        """Never answered."""
        raise NotImplementedError

    def answer(self, query: models.AnswerQuery) -> models.AnswerResult:
        """Never answered."""
        raise NotImplementedError

    def get_scene(self, scene_set_version: str, index: int, caller: models.Caller) -> Json:
        """Never answered."""
        raise NotImplementedError

    def get_keyframe(self, digest: str, caller: models.Caller) -> models.DerivativeContent:
        """Never answered."""
        raise NotImplementedError
