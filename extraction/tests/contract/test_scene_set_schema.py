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


class TestGivenSampleReadByEnrichment:
    """The consumer step decodes the producer's sample through its own view."""

    def test_when_decoded_then_scenes_and_derivatives_visible(self) -> None:
        from scene_rag.extraction.enrichment.scene_store import scene_set_from_document

        view = scene_set_from_document(dict(SAMPLE))

        assert view.version == SAMPLE["_id"]
        assert view.asset_version == SAMPLE["asset_version"]
        assert [scene.index for scene in view.scenes] == [0, 1]
        assert len(view.scenes[0].keyframes) == 2
        assert (
            view.scenes[0].keyframes[0].derivative.digest
            == (SAMPLE["scenes"][0]["keyframes"][0]["derivative"]["digest"])
        )
        assert view.scenes[1].audio_clip is not None
        assert view.scenes[1].audio_clip.media_type == "audio/wav"
