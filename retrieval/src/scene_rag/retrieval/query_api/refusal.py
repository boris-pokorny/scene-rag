"""When the role declines to answer: the refusal policy.

Part of the ``query_config`` id: the threshold decides whether a question gets an answer.
"""

from collections.abc import Sequence

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.config import RefusalConfig


class ScoreThresholdPolicy:
    """`RefusalPolicy` refusing unless some source reaches the configured similarity."""

    def __init__(self, config: RefusalConfig) -> None:
        """Bind to the threshold."""
        self._min_score = config.min_score

    def should_refuse(self, sources: Sequence[models.Source]) -> bool:
        """Return whether no source scores at least the threshold."""
        return not any(source.score >= self._min_score for source in sources)
