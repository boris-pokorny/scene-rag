"""The decoder of query-api's published responses, pinned by retrieval's own samples.

`samples/` are copies of `retrieval/contracts/published/samples/` (2026-10-30); they are
updated by hand when retrieval publishes a new major of the query API.
"""

from scene_rag.evaluation.models import Served, Source
from scene_rag.evaluation.query_api_client import (
    answer_response_from_json,
    search_response_from_json,
)
from tests.test_query_api_client import sample

VERSION = "sha256:88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"
SERVED = Served("11d0a9e2c4f1", "5e2d4c6b8a90", None, "chunks_5e2d4c6b8a90")


class TestGivenSearchResponseSample:
    """The published search response."""

    def test_when_decoded_then_sources_in_rank_order_and_served(self) -> None:
        response = search_response_from_json(sample("search-response"))

        assert response.sources == (Source(VERSION, 4.0, 8.0), Source(VERSION, 0.0, 4.0))
        assert response.served == SERVED and response.answer is None


class TestGivenAnswerResponseSample:
    """The published answer response."""

    def test_when_decoded_then_generation_fields_there(self) -> None:
        response = answer_response_from_json(sample("answer-response"))

        assert response.answer is not None
        assert response.answer.text.endswith("[1].") and not response.answer.refused
        assert response.answer.citations == (1,) and response.answer.invalid_citations == ()
        assert (response.answer.prompt_tokens, response.answer.completion_tokens) == (412, 23)
        assert len(response.sources) == 2 and response.served == SERVED
