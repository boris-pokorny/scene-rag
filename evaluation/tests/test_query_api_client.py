"""The query-api client over a fake transport."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from scene_rag.evaluation.http_transport import HttpStatusError, Json, TransportError
from scene_rag.evaluation.protocols import QueryApiError
from scene_rag.evaluation.query_api_client import (
    ENTITLEMENTS_HEADER,
    QueryApiClient,
    answer_response_from_json,
)

SAMPLES = Path(__file__).resolve().parent / "contract" / "samples"


def sample(name: str) -> Json:
    document: Json = json.loads((SAMPLES / f"{name}.json").read_text())
    return document


class RecordingTransport:
    """Answers one document, or raises; records every request."""

    def __init__(self, answer: Json | Exception) -> None:
        self.answer = answer
        self.requests: list[tuple[str, str, Json | None, dict[str, str], float]] = []

    def exchange(
        self,
        method: str,
        url: str,
        payload: Json | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Json:
        self.requests.append((method, url, payload, dict(headers), timeout))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class TestGivenQueryApi:
    """Requests carry the question, the count and the caller's entitlements."""

    def test_when_searched_then_posted_with_top_k_and_header(self) -> None:
        transport = RecordingTransport(sample("search-response"))
        client = QueryApiClient("http://q:8081/", transport, 7.0)

        response = client.search("When?", frozenset({"b", "a"}), 10)

        assert transport.requests == [
            (
                "POST",
                "http://q:8081/search",
                {"question": "When?", "top_k": 10},
                {ENTITLEMENTS_HEADER: "a,b"},
                7.0,
            )
        ]
        assert len(response.sources) == 2 and response.answer is None

    def test_when_answered_then_posted_without_count(self) -> None:
        transport = RecordingTransport(sample("answer-response"))

        response = QueryApiClient("http://q", transport, 1.0).answer("When?", frozenset())

        assert transport.requests[0][2] == {"question": "When?"}
        assert transport.requests[0][3] == {ENTITLEMENTS_HEADER: ""}
        assert response.answer is not None and response.answer.citations == (1,)

    def test_when_health_checked_then_got(self) -> None:
        transport = RecordingTransport({"status": "ok"})

        QueryApiClient("http://q", transport, 1.0).health()

        assert transport.requests[0][:2] == ("GET", "http://q/health")

    @pytest.mark.parametrize(
        "error", [HttpStatusError(503, "http://q/search", "busy"), TransportError("down")]
    )
    def test_when_transport_fails_then_query_api_error(self, error: Exception) -> None:
        client = QueryApiClient("http://q", RecordingTransport(error), 1.0)

        with pytest.raises(QueryApiError):
            client.search("When?", frozenset({"a"}), 5)
        with pytest.raises(QueryApiError):
            client.health()

    def test_when_answer_undecodable_then_query_api_error(self) -> None:
        client = QueryApiClient("http://q", RecordingTransport({"answer": "x"}), 1.0)

        with pytest.raises(QueryApiError, match="undecodable answer"):
            client.answer("When?", frozenset({"a"}))

    def test_when_search_undecodable_then_query_api_error(self) -> None:
        client = QueryApiClient("http://q", RecordingTransport({"sources": [{}]}), 1.0)

        with pytest.raises(QueryApiError, match="undecodable search"):
            client.search("When?", frozenset({"a"}), 5)


class TestGivenUsage:
    """Token counts are optional, and must be integers when given."""

    def test_when_refused_then_no_usage(self) -> None:
        document: dict[str, Any] = {**sample("answer-response"), "usage": None, "refused": True}

        answer = answer_response_from_json(document).answer

        assert answer is not None and answer.prompt_tokens is None and answer.refused

    def test_when_count_not_integer_then_rejected(self) -> None:
        document = sample("answer-response")
        document["usage"] = {"prompt_tokens": "many", "completion_tokens": None}

        with pytest.raises(QueryApiError):
            answer_response_from_json(document)
