"""The score threshold policy."""

import pytest

from scene_rag.retrieval.query_api.config import RefusalConfig
from scene_rag.retrieval.query_api.refusal import ScoreThresholdPolicy
from tests.support.query_fakes import make_entry_view, make_source

POLICY = ScoreThresholdPolicy(RefusalConfig(min_score=0.5))


class TestGivenThreshold:
    """A threshold of one half."""

    @pytest.mark.parametrize(
        "scores, refused",
        [((), True), ((0.49,), True), ((0.5,), False), ((0.1, 0.9), False), ((0.2, 0.3), True)],
    )
    def test_when_best_score_compared_then_refused_below(
        self, scores: tuple[float, ...], refused: bool
    ) -> None:
        sources = [make_source(make_entry_view(i), s) for i, s in enumerate(scores)]

        assert POLICY.should_refuse(sources) is refused
