"""The Chunk Store documents match contracts/internal/*.schema.json."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scene_rag.retrieval.chunking.chunk_store import (
    chunk_from_document,
    chunk_set_from_document,
    chunk_set_to_document,
    chunk_to_document,
)
from tests.support.chunking_fakes import make_chunk, make_chunk_set, make_records

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "internal"


def as_json(document: dict[str, Any]) -> dict[str, Any]:
    """Render a store document the way the samples are written: dates as RFC 3339 strings."""

    def convert(value: Any) -> Any:  # noqa: ANN401 - walks untyped JSON
        if isinstance(value, datetime):
            return value.isoformat().replace("+00:00", "Z")
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    result: dict[str, Any] = convert(document)
    return result


def load(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((CONTRACTS / name).read_text())
    return document


def validator(name: str) -> jsonschema.Draft202012Validator:
    schema = load(f"{name}.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["$id"].endswith(f"/retrieval/internal/{name}.schema.json")
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


class TestGivenChunkSchema:
    """The chunk schema, its sample and the encoder."""

    def test_when_sample_validated_then_accepted(self) -> None:
        validator("chunk").validate(load("samples/chunk.json"))

    def test_when_records_encoded_then_valid(self) -> None:
        records = make_records()
        for index in range(len(records)):
            validator("chunk").validate(as_json(chunk_to_document(make_chunk(records, index))))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        sample = load("samples/chunk.json")
        document = dict(sample)
        document["created_at"] = datetime.fromisoformat(document["created_at"])

        chunk = chunk_from_document(document)

        assert chunk.version == sample["_id"]
        assert chunk.lineage.step == "chunking"
        assert as_json(chunk_to_document(chunk)) == sample

    @pytest.mark.parametrize("field", ["text", "scene_indices"])
    def test_when_required_content_empty_then_rejected(self, field: str) -> None:
        document = as_json(chunk_to_document(make_chunk()))
        document[field] = "" if field == "text" else []

        with pytest.raises(jsonschema.ValidationError):
            validator("chunk").validate(document)


class TestGivenChunkSetSchema:
    """The chunk set schema, its sample and the encoder."""

    def test_when_sample_validated_then_accepted(self) -> None:
        validator("chunk-set").validate(load("samples/chunk-set.json"))

    @pytest.mark.parametrize("chunk_count", [0, 3])
    def test_when_record_encoded_then_valid(self, chunk_count: int) -> None:
        document = chunk_set_to_document(make_chunk_set(chunk_count=chunk_count))

        validator("chunk-set").validate(as_json(document))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        sample = load("samples/chunk-set.json")
        document = dict(sample)
        document["created_at"] = datetime.fromisoformat(document["created_at"])

        chunk_set = chunk_set_from_document(document)

        assert chunk_set.version == sample["_id"]
        assert [str(pin) for pin in chunk_set.pins] == sample["pins"]
        assert as_json(chunk_set_to_document(chunk_set)) == sample
