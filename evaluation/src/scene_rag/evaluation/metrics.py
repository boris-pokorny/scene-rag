"""The shipped metrics, one `Metric` each.

Retrieval metrics score searches of answerable cases only; leakage scores every case, since
an unanswerable question can still leak; generation metrics score answers deterministically.
Faithfulness and correctness need a `Judge` and are not shipped yet.
"""

import dataclasses
import math
from typing import Literal

from scene_rag.evaluation.config import UseCase
from scene_rag.evaluation.models import Aggregation, Answer, Case, Observation
from scene_rag.evaluation.relevance import covered, gains, overlaps


def _answer_of(observation: Observation) -> Answer | None:
    return observation.response.answer


@dataclasses.dataclass(frozen=True)
class HitAtK:
    """1 when a relevant source is among the first ``k``."""

    k: int
    use_case: UseCase = "search"
    aggregation: Aggregation = "mean"

    @property
    def name(self) -> str:
        """Return ``hit@<k>``."""
        return f"hit@{self.k}"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Score one search; ``None`` for an unanswerable case."""
        if not case.answerable:
            return None
        return 1.0 if covered(observation.response.sources[: self.k], case.relevant) else 0.0


@dataclasses.dataclass(frozen=True)
class RecallAtK:
    """Share of the labelled spans the first ``k`` sources overlap."""

    k: int
    use_case: UseCase = "search"
    aggregation: Aggregation = "mean"

    @property
    def name(self) -> str:
        """Return ``recall@<k>``."""
        return f"recall@{self.k}"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Score one search; ``None`` for an unanswerable case."""
        if not case.answerable:
            return None
        found = covered(observation.response.sources[: self.k], case.relevant)
        return len(found) / len(case.relevant)


@dataclasses.dataclass(frozen=True)
class ReciprocalRank:
    """1 / rank of the first relevant source; 0 when none is returned. Its mean is MRR."""

    use_case: UseCase = "search"
    aggregation: Aggregation = "mean"

    @property
    def name(self) -> str:
        """Return ``mrr``."""
        return "mrr"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Score one search; ``None`` for an unanswerable case."""
        if not case.answerable:
            return None
        for rank, source in enumerate(observation.response.sources, start=1):
            if any(overlaps(source, span) for span in case.relevant):
                return 1.0 / rank
        return 0.0


@dataclasses.dataclass(frozen=True)
class NdcgAtK:
    """Normalised discounted cumulative gain of the first ``k`` sources, binary gains."""

    k: int
    use_case: UseCase = "search"
    aggregation: Aggregation = "mean"

    @property
    def name(self) -> str:
        """Return ``ndcg@<k>``."""
        return f"ndcg@{self.k}"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Score one search; ``None`` for an unanswerable case."""
        if not case.answerable:
            return None
        ranked = gains(observation.response.sources[: self.k], case.relevant)
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ranked))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(self.k, len(case.relevant))))
        return dcg / ideal


@dataclasses.dataclass(frozen=True)
class Leakage:
    """Sources overlapping a span the caller must not see; summed, it must be zero."""

    use_case: UseCase
    aggregation: Aggregation = "sum"

    @property
    def name(self) -> str:
        """Return ``leakage``."""
        return "leakage"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Count the leaking sources of one observation."""
        return float(
            sum(
                any(overlaps(source, span) for span in case.forbidden)
                for source in observation.response.sources
            )
        )


@dataclasses.dataclass(frozen=True)
class CitationValidity:
    """1 when every number the answer cites names a source."""

    use_case: UseCase = "answer"
    aggregation: Aggregation = "mean"

    @property
    def name(self) -> str:
        """Return ``citation_validity``."""
        return "citation_validity"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Score one answer; ``None`` for a refusal, which cites nothing."""
        answer = _answer_of(observation)
        if answer is None or answer.refused:
            return None
        return 0.0 if answer.invalid_citations else 1.0


@dataclasses.dataclass(frozen=True)
class RefusalCorrectness:
    """1 when the answer refuses exactly the unanswerable questions."""

    use_case: UseCase = "answer"
    aggregation: Aggregation = "mean"

    @property
    def name(self) -> str:
        """Return ``refusal_correctness``."""
        return "refusal_correctness"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Score one answer."""
        answer = _answer_of(observation)
        if answer is None:
            return None
        return 1.0 if answer.refused != case.answerable else 0.0


@dataclasses.dataclass(frozen=True)
class Latency:
    """Wall-clock time of the call in milliseconds; rolls up to p50 and p95."""

    use_case: UseCase
    aggregation: Aggregation = "percentiles"

    @property
    def name(self) -> str:
        """Return ``latency_ms``."""
        return "latency_ms"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Return the call's latency."""
        return observation.latency_ms


@dataclasses.dataclass(frozen=True)
class Tokens:
    """Tokens the generation consumed, as the model reports them; summed."""

    kind: Literal["prompt", "completion"]
    use_case: UseCase = "answer"
    aggregation: Aggregation = "sum"

    @property
    def name(self) -> str:
        """Return ``prompt_tokens`` or ``completion_tokens``."""
        return f"{self.kind}_tokens"

    def score(self, case: Case, observation: Observation) -> float | None:
        """Return the count; ``None`` when the model did not report it or nothing ran."""
        answer = _answer_of(observation)
        if answer is None:
            return None
        count = answer.prompt_tokens if self.kind == "prompt" else answer.completion_tokens
        return None if count is None else float(count)
