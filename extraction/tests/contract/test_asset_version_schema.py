"""The registry document encoding matches contracts/internal/asset-version.schema.json."""

import json
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest
from support.fakes import make_record, make_reference, sha256_version

from scene_rag.extraction.ingestion.registry import from_document, to_document

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "internal"
SCHEMA = json.loads((CONTRACTS / "asset-version.schema.json").read_text())
SAMPLE = json.loads((CONTRACTS / "samples" / "asset-version.json").read_text())


def as_json(value: object) -> object:
    """Render BSON-bound values (datetimes) the way the JSON samples do."""
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [as_json(item) for item in value]
    return value


class TestGivenSchema:
    """The checked JSON Schema with format validation on."""

    @pytest.fixture
    def validator(self) -> jsonschema.Draft202012Validator:
        jsonschema.Draft202012Validator.check_schema(SCHEMA)
        return jsonschema.Draft202012Validator(
            SCHEMA, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
        )

    def test_when_sample_validated_then_accepted(
        self, validator: jsonschema.Draft202012Validator
    ) -> None:
        validator.validate(SAMPLE)

    def test_when_live_record_encoded_then_valid(
        self, validator: jsonschema.Draft202012Validator
    ) -> None:
        record = make_record(
            sha256_version(b"x"),
            make_reference("asset-0000000000000001"),
            make_reference("asset-0000000000000002", source_id="library"),
        )
        validator.validate(as_json(to_document(record)))

    def test_when_tombstone_encoded_then_valid(
        self, validator: jsonschema.Draft202012Validator
    ) -> None:
        record = make_record(sha256_version(b"x"), retired_at=make_reference().seen_at)
        document = to_document(record)
        document["sources"] = []
        validator.validate(as_json(document))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        document = dict(SAMPLE)
        for key in ("first_seen_at", "retired_at"):
            if document[key] is not None:
                document[key] = datetime.fromisoformat(document[key])
        document["sources"] = [
            {
                **source,
                "modified_at": datetime.fromisoformat(source["modified_at"]),
                "seen_at": datetime.fromisoformat(source["seen_at"]),
            }
            for source in document["sources"]
        ]

        record = from_document(document)

        assert record.version == SAMPLE["_id"]
        assert record.entitlements == frozenset({"public", "team-platform"})
        assert as_json(to_document(record)) == SAMPLE
