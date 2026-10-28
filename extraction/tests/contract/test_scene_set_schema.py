"""The scene set encoding matches contracts/internal/scene-set.schema.json."""

import json
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest

from scene_rag.extraction.segmentation.scene_store import from_document, to_document
from tests.contract.test_asset_version_schema import as_json
from tests.support.segmentation_fakes import make_scene_set

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "internal"
SCHEMA = json.loads((CONTRACTS / "scene-set.schema.json").read_text())
SAMPLE = json.loads((CONTRACTS / "samples" / "scene-set.json").read_text())


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

    def test_when_record_encoded_then_valid(
        self, validator: jsonschema.Draft202012Validator
    ) -> None:
        validator.validate(as_json(to_document(make_scene_set())))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        document = dict(SAMPLE)
        document["created_at"] = datetime.fromisoformat(document["created_at"])

        record = from_document(document)

        assert record.version == SAMPLE["_id"]
        assert record.lineage.step == "segmentation"
        assert as_json(to_document(record)) == SAMPLE
