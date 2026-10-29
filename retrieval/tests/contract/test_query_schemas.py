"""The published JSON Schemas of the query-api, their samples, and the encoder against them."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from referencing import Registry, Resource

from scene_rag.retrieval.query_api import encoding, models
from scene_rag.retrieval.query_api.app import create_app
from scene_rag.retrieval.query_api.mcp_server import create_mcp_server
from tests.query_api.test_encoding import ANSWER, SEARCH
from tests.support.query_fakes import FakeQueryService, make_entry_view, make_source

PUBLISHED = Path(__file__).resolve().parents[2] / "contracts" / "published"
SCHEMAS = [
    "search-request",
    "answer-request",
    "search-response",
    "answer-response",
    "source",
    "served",
    "error",
]


def load_schema(name: str) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads((PUBLISHED / f"{name}.schema.json").read_text())
    return schema


REGISTRY = Registry().with_resources(
    (load_schema(name)["$id"], Resource.from_contents(load_schema(name))) for name in SCHEMAS
)


def validator(name: str) -> jsonschema.Draft202012Validator:
    schema = load_schema(name)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema, registry=REGISTRY, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


class TestGivenSchemas:
    """Every published schema is valid and strict; every sample validates."""

    @pytest.mark.parametrize("name", SCHEMAS)
    def test_when_checked_then_valid_and_closed(self, name: str) -> None:
        schema = load_schema(name)

        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        assert schema["$id"].endswith(f"/retrieval/published/{name}.schema.json")

    @pytest.mark.parametrize("name", SCHEMAS)
    def test_when_sample_validated_then_accepted(self, name: str) -> None:
        sample = json.loads((PUBLISHED / "samples" / f"{name}.json").read_text())

        validator(name).validate(sample)

    def test_when_error_codes_listed_then_the_ones_the_app_answers(self) -> None:
        codes = set(load_schema("error")["properties"]["error"]["properties"]["code"]["enum"])
        source = (
            Path(create_app.__code__.co_filename).read_text()
            + Path(create_mcp_server.__code__.co_filename).read_text()
        )

        # `method_not_allowed` is derived from the status phrase, never spelled out.
        assert codes - {"method_not_allowed"} == {c for c in codes if f'"{c}"' in source}
        assert {"invalid_parameter", "not_found", "method_not_allowed", "internal_error"} <= codes


class TestGivenEncoder:
    """What the encoder produces validates against the schemas."""

    def test_when_results_encoded_then_valid(self) -> None:
        validator("search-response").validate(encoding.encode_search_result(SEARCH))
        validator("answer-response").validate(encoding.encode_answer_result(ANSWER))
        validator("source").validate(encoding.encode_source(make_source(make_entry_view(3), 0.0)))

    def test_when_refusal_encoded_then_valid(self) -> None:
        refusal = models.AnswerResult(
            "I cannot answer that from the indexed videos.",
            True,
            (),
            (),
            (),
            models.NO_USAGE,
            SEARCH.served,
        )

        validator("answer-response").validate(encoding.encode_answer_result(refusal))

    def test_when_empty_search_encoded_then_valid(self) -> None:
        validator("search-response").validate(
            encoding.encode_search_result(models.SearchResult((), SEARCH.served))
        )

    @pytest.mark.parametrize("name", ["search-request", "answer-request"])
    def test_when_request_minimal_then_valid_and_blank_rejected(self, name: str) -> None:
        validator(name).validate({"question": "q"})

        with pytest.raises(jsonschema.ValidationError):
            validator(name).validate({"question": ""})
        with pytest.raises(jsonschema.ValidationError):
            validator(name).validate({"question": "q", "top_k": 0})


class TestGivenFakeService:
    """The fake the adapter tests run over satisfies the protocol's shapes."""

    def test_when_served_then_valid(self) -> None:
        validator("served").validate(encoding.encode_served(FakeQueryService().served))
