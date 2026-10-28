"""The enrichment document encoding matches contracts/internal/enrichment.schema.json."""

import json
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest

from scene_rag.extraction.enrichment.enrichment_store import from_document, to_document
from scene_rag.extraction.enrichment.models import SemanticSegment
from tests.contract.test_asset_version_schema import as_json
from tests.support.enrichment_fakes import make_enrichment, make_scene_set

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "internal"
SCHEMA = json.loads((CONTRACTS / "enrichment.schema.json").read_text())
SAMPLE = json.loads((CONTRACTS / "samples" / "enrichment.json").read_text())


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

    @pytest.mark.parametrize(
        "segments",
        [
            (),
            (SemanticSegment(0.0, 4.0, "A talk."),),
            (
                SemanticSegment(0.5, 1.5, "hello", "en", 0.9),
                SemanticSegment(1.5, 4.0, "there", "en", None),
            ),
        ],
    )
    def test_when_record_encoded_then_valid(
        self,
        validator: jsonschema.Draft202012Validator,
        segments: tuple[SemanticSegment, ...],
    ) -> None:
        record = make_enrichment(make_scene_set(), 0, "transcript", segments=segments)

        validator.validate(as_json(to_document(record)))

    def test_when_sample_decoded_and_encoded_then_identical(self) -> None:
        document = dict(SAMPLE)
        document["created_at"] = datetime.fromisoformat(document["created_at"])

        record = from_document(document)

        assert record.version == SAMPLE["_id"]
        assert record.lineage.step == "enrichment"
        assert record.enricher == "transcript"
        assert as_json(to_document(record)) == SAMPLE
