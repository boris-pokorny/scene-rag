"""The domain model: frozen values with the few rules they carry."""

import dataclasses

import pytest

from scene_rag.retrieval.query_api import models


class TestGivenValues:
    """Values of the model."""

    def test_when_anonymous_then_no_entitlements(self) -> None:
        assert models.ANONYMOUS.entitlements == frozenset()
        assert models.Usage(None, None) == models.NO_USAGE

    def test_when_mutated_then_frozen(self) -> None:
        query = models.SearchQuery(question="q", caller=models.ANONYMOUS)

        with pytest.raises(dataclasses.FrozenInstanceError):
            query.question = "other"  # type: ignore[misc]

    def test_when_query_built_then_defaults_apply(self) -> None:
        query = models.AnswerQuery(question="q", caller=models.ANONYMOUS)

        assert query.top_k is None
        assert query.filters == models.SearchFilters()
