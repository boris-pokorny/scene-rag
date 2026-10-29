"""Hyperparameters of the query-api role.

The field defaults *are* the values: changing one is a code change that changes the
``query_config`` id every answer reports, and so starts a candidate release. Nothing here
comes from the environment; connection settings, timeouts, host and port live in
`settings.py`. The query-time embedder is deliberately absent: the role takes it from the
index it serves, so index and query can never embed differently. The served index is pinned
here, as the steps pin their upstream fingerprints.
"""

import dataclasses
import math
import re
from typing import Final, Literal

FINGERPRINT_PATTERN: Final = re.compile(r"^[0-9a-f]{12}$")

type RerankerName = Literal["none"]
"""Rerankers the role knows; ``none`` keeps the retriever's order."""

RERANKERS: Final[frozenset[str]] = frozenset({"none"})


@dataclasses.dataclass(frozen=True)
class UpstreamConfig:
    """Which index the role serves.

    Attributes:
        embedding_fingerprint: Serve the index built under this embedding fingerprint; the
            newest index in the ledger when ``None``. A release pins it so a rollback serves
            exactly what was gated.
    """

    embedding_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When the pin is not twelve hex digits.
        """
        fingerprint = self.embedding_fingerprint
        if fingerprint is not None and not FINGERPRINT_PATTERN.match(fingerprint):
            raise ValueError("embedding_fingerprint must be twelve hex digits or None")


@dataclasses.dataclass(frozen=True)
class RetrieverConfig:
    """Tunables of retrieval.

    Attributes:
        top_k: Sources returned when a request names no count.
        max_top_k: Most sources a request may ask for; a larger request is capped.
        max_question_chars: Longest question accepted.
    """

    top_k: int = 5
    max_top_k: int = 20
    max_question_chars: int = 2000

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working retriever.
        """
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.max_top_k < self.top_k:
            raise ValueError("max_top_k must be at least top_k")
        if self.max_question_chars < 1:
            raise ValueError("max_question_chars must be at least 1")


@dataclasses.dataclass(frozen=True)
class RerankerConfig:
    """Which reranker reorders the retrieved sources.

    Attributes:
        name: The reranker; ``none`` keeps the retriever's order.
    """

    name: RerankerName = "none"

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When the reranker is not one the role knows.
        """
        if self.name not in RERANKERS:
            raise ValueError(f"reranker must be one of {sorted(RERANKERS)}, not {self.name!r}")


@dataclasses.dataclass(frozen=True)
class RefusalConfig:
    """When the role declines to answer, and what it says then.

    Attributes:
        min_score: Lowest similarity a source must reach for the question to be answered;
            below it for every source, the answer is a refusal and no model is called.
        message: The refusal.
    """

    min_score: float = 0.35
    message: str = "I cannot answer that from the indexed videos."

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working policy.
        """
        if not math.isfinite(self.min_score):
            raise ValueError("min_score must be a finite number")
        if not self.message:
            raise ValueError("refusal message must not be empty")


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    """The prompt the generator answers under.

    Attributes:
        system_template: The system message.
        user_template: The user message; ``{sources}`` and ``{question}`` are filled in.
        source_template: One line per source inside ``{sources}``; ``{number}``,
            ``{filename}``, ``{start}``, ``{end}`` and ``{text}`` are filled in.
    """

    system_template: str = (
        "You answer questions about a video library from the numbered sources below. Use "
        "only the sources. Cite the source of every claim by its number in square brackets, "
        "like [2]. If the sources do not answer the question, say so."
    )
    user_template: str = "Sources:\n{sources}\n\nQuestion: {question}"
    source_template: str = "[{number}] {filename} {start:.1f}-{end:.1f}s: {text}"

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a template lacks a placeholder the role fills in.
        """
        for name in ("{sources}", "{question}"):
            if name not in self.user_template:
                raise ValueError(f"user_template must contain {name}")
        for name in ("{number}", "{text}"):
            if name not in self.source_template:
                raise ValueError(f"source_template must contain {name}")


@dataclasses.dataclass(frozen=True)
class GeneratorConfig:
    """Tunables of the answer generator.

    Attributes:
        name: Generator model, as the backend names it; the ``query_config`` id pins its
            digest.
        temperature: Sampling temperature; zero for the most likely answer.
        seed: Sampling seed, so a temperature above zero still repeats.
        max_tokens: Longest answer, in tokens.
        num_ctx: Context the model runs with, in tokens; prompt and answer must fit.
    """

    name: str = "qwen2.5:3b"
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 512
    num_ctx: int = 8192

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot define a working generator.
        """
        if not self.name:
            raise ValueError("generator name must not be empty")
        if not (math.isfinite(self.temperature) and self.temperature >= 0):
            raise ValueError("temperature must be a finite number of at least 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.num_ctx < 1:
            raise ValueError("num_ctx must be at least 1")


@dataclasses.dataclass(frozen=True)
class QueryConfig:
    """Every tunable of the query-api role, and nothing else.

    Attributes:
        upstream: Which index is served.
        retriever: Tunables of retrieval.
        reranker: Which reranker reorders the sources.
        refusal: When and how the role declines.
        prompt: The prompt the generator answers under.
        generator: Tunables of the answer generator.
    """

    upstream: UpstreamConfig = UpstreamConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    refusal: RefusalConfig = RefusalConfig()
    prompt: PromptConfig = PromptConfig()
    generator: GeneratorConfig = GeneratorConfig()
