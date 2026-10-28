"""The published JSON Schemas of the scene-api, their samples, and the encoder against them."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scene_rag.extraction.scene_api import encoding, models
from tests.support.scene_api_fakes import (
    T0,
    make_asset_view,
    make_record,
    make_scene_set_view,
    make_source,
    retired,
    sha256_version,
)

PUBLISHED = Path(__file__).resolve().parents[2] / "contracts" / "published"
SCHEMAS = ["scene-record", "scene-set-summary", "asset-version", "page", "error"]
SAMPLES = ["scene-record", "scene-set-summary", "asset-version", "error"]


def load_schema(name: str) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads((PUBLISHED / f"{name}.schema.json").read_text())
    return schema


def validator(name: str) -> jsonschema.Draft202012Validator:
    schema = load_schema(name)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


class TestGivenSchemas:
    """Every published schema is valid and strict; every sample validates."""

    @pytest.mark.parametrize("name", SCHEMAS)
    def test_when_checked_then_valid_and_closed(self, name: str) -> None:
        schema = load_schema(name)

        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        assert schema["$id"].endswith(f"/published/{name}.schema.json")

    @pytest.mark.parametrize("name", SAMPLES)
    def test_when_sample_validated_then_accepted(self, name: str) -> None:
        sample = json.loads((PUBLISHED / "samples" / f"{name}.json").read_text())

        validator(name).validate(sample)

    def test_when_sample_record_version_recomputed_then_matches(self) -> None:
        sample = json.loads((PUBLISHED / "samples" / "scene-record.json").read_text())

        assert sample["version"] == models.scene_record_version(
            sample["scene_set_version"],
            sample["scene"]["index"],
            [item["version"] for item in sample["enrichments"]],
        )


class TestGivenEncoder:
    """What the encoder produces validates against the schemas."""

    def test_when_record_encoded_then_valid(self) -> None:
        asset = make_asset_view(
            sha256_version(b"abcd"),
            make_source(),
            make_source("asset-0000000000000002", "library", "talk.mp4"),
        )
        for index in range(3):
            record = make_record(make_scene_set_view(), index, asset=asset)
            validator("scene-record").validate(encoding.encode_scene_record(record))

    def test_when_record_without_enrichments_or_clip_then_valid(self) -> None:
        record = make_record(make_scene_set_view(), 1, enrichments=())

        validator("scene-record").validate(encoding.encode_scene_record(record))

    def test_when_retired_record_encoded_then_valid(self) -> None:
        record = make_record(asset=retired(make_asset_view()))

        validator("scene-record").validate(encoding.encode_scene_record(record))

    def test_when_summary_encoded_then_valid(self) -> None:
        summary = models.SceneSetSummary(
            version="sha256:ab",
            asset_version="sha256:cd",
            segmentation_fingerprint="0123456789ab",
            scene_count=3,
            created_at=T0,
            live=True,
            enrichments=(models.EnrichmentCoverage("transcript", "e41a9c27b0d5", 3, True),),
        )

        validator("scene-set-summary").validate(encoding.encode_scene_set_summary(summary))

    @pytest.mark.parametrize("live", [True, False])
    def test_when_asset_encoded_then_valid(self, live: bool) -> None:
        asset = make_asset_view() if live else retired(make_asset_view())

        validator("asset-version").validate(encoding.encode_asset_version(asset))

    def test_when_page_and_error_encoded_then_valid(self) -> None:
        validator("page").validate(encoding.encode_page(models.Page((), None), dict))
        validator("page").validate(encoding.encode_page(models.Page(("x",), "c"), lambda v: {}))
        validator("error").validate(encoding.encode_error("not_found", "gone"))
