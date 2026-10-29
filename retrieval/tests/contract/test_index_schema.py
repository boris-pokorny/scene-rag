"""The index ledger documents and the point payload match contracts/internal/*.schema.json."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scene_rag.retrieval.embedding.index_store import (
    index_from_document,
    index_set_from_document,
    index_set_to_document,
    index_to_document,
)
from scene_rag.retrieval.embedding.qdrant_index import entry_from_payload, entry_to_payload
from tests.support.embedding_fakes import (
    as_document,
    make_chunk_views,
    make_entry,
    make_index,
    make_index_set,
)

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "internal"


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


class TestGivenIndexEntrySchema:
    """The payload schema, its sample and the encoder."""

    def test_when_sample_validated_then_accepted(self) -> None:
        validator("index-entry").validate(load("samples/index-entry.json"))

    def test_when_chunks_encoded_then_valid(self) -> None:
        for chunk in make_chunk_views():
            validator("index-entry").validate(entry_to_payload(make_entry(chunk)))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        sample = load("samples/index-entry.json")

        entry = entry_from_payload(sample)

        assert entry.version == sample["version"]
        assert entry.lineage.step == "embedding"
        assert entry_to_payload(entry) == sample

    @pytest.mark.parametrize("field", ["text", "scene_indices"])
    def test_when_required_content_empty_then_rejected(self, field: str) -> None:
        payload = entry_to_payload(make_entry(make_chunk_views()[0]))
        payload[field] = "" if field == "text" else []

        with pytest.raises(jsonschema.ValidationError):
            validator("index-entry").validate(payload)


class TestGivenIndexSetSchema:
    """The index set schema, its sample and the encoder."""

    def test_when_sample_validated_then_accepted(self) -> None:
        validator("index-set").validate(load("samples/index-set.json"))

    @pytest.mark.parametrize("count", [0, 3])
    def test_when_record_encoded_then_valid(self, count: int) -> None:
        document = index_set_to_document(make_index_set(make_chunk_views(count=count)))

        validator("index-set").validate(as_document(document))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        sample = load("samples/index-set.json")
        document = dict(sample)
        document["created_at"] = datetime.fromisoformat(document["created_at"])

        index_set = index_set_from_document(document)

        assert index_set.version == sample["_id"]
        assert as_document(index_set_to_document(index_set)) == sample


class TestGivenIndexSchema:
    """The index record schema, its sample and the encoder."""

    def test_when_sample_validated_then_accepted(self) -> None:
        validator("index").validate(load("samples/index.json"))

    def test_when_record_encoded_then_valid(self) -> None:
        validator("index").validate(as_document(index_to_document(make_index())))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        sample = load("samples/index.json")
        document = dict(sample)
        document["created_at"] = datetime.fromisoformat(document["created_at"])

        index = index_from_document(document)

        assert index.fingerprint == sample["_id"]
        assert as_document(index_to_document(index)) == sample

    def test_when_distance_unknown_then_rejected(self) -> None:
        document = as_document(index_to_document(make_index()))
        document["distance"] = "manhattan"

        with pytest.raises(jsonschema.ValidationError):
            validator("index").validate(document)
