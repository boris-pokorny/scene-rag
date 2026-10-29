"""The encoder produces exactly the published samples."""

import json
from pathlib import Path
from typing import Any

from scene_rag.retrieval.query_api import encoding, models
from tests.support.query_fakes import SERVED, make_entry_view, make_source

SAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "published" / "samples"
SOURCES = (make_source(make_entry_view(1), 0.91), make_source(make_entry_view(0), 0.63))
SEARCH = models.SearchResult(sources=SOURCES, served=SERVED)
ANSWER = models.AnswerResult(
    answer="The pipeline cuts the video into scenes right after the introduction [1].",
    refused=False,
    citations=(1,),
    invalid_citations=(),
    sources=SOURCES,
    usage=models.Usage(412, 23),
    served=SERVED,
)


def sample(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((SAMPLES / f"{name}.json").read_text())
    return document


class TestGivenResults:
    """The results the samples were rendered from."""

    def test_when_source_encoded_then_sample(self) -> None:
        assert encoding.encode_source(SOURCES[0]) == sample("source")

    def test_when_served_encoded_then_sample(self) -> None:
        assert encoding.encode_served(SERVED) == sample("served")

    def test_when_search_result_encoded_then_sample(self) -> None:
        assert encoding.encode_search_result(SEARCH) == sample("search-response")

    def test_when_answer_result_encoded_then_sample(self) -> None:
        assert encoding.encode_answer_result(ANSWER) == sample("answer-response")

    def test_when_error_encoded_then_sample(self) -> None:
        assert encoding.encode_error("invalid_parameter", "question must not be blank") == sample(
            "error"
        )

    def test_when_health_encoded_then_status_and_served(self) -> None:
        assert encoding.encode_health(SERVED) == {"status": "ok", "served": sample("served")}

    def test_when_refusal_encoded_then_null_usage(self) -> None:
        refusal = models.AnswerResult("no", True, (), (), (), models.NO_USAGE, SERVED)

        document = encoding.encode_answer_result(refusal)

        assert document["usage"] == {"prompt_tokens": None, "completion_tokens": None}
        assert document["refused"] is True
