"""The decoder of scene-api's published documents, pinned by extraction's own samples.

`samples/` are copies of `extraction/contracts/published/samples/` (2026-10-28); they are
updated by hand when extraction publishes a new major of the scene record contract.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.protocols import SceneApiError
from scene_rag.retrieval.chunking.scene_api_client import (
    asset_version_from_json,
    scene_record_from_json,
    summary_from_json,
)
from tests.support.chunking_fakes import encode_scene_record, encode_summary

SAMPLES = Path(__file__).resolve().parent / "samples"


def load(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((SAMPLES / f"{name}.json").read_text())
    return document


class TestGivenSceneRecordSample:
    """The published scene record."""

    def test_when_decoded_then_every_field_the_step_uses_is_there(self) -> None:
        record = scene_record_from_json(load("scene-record"))

        assert record.version == load("scene-record")["version"]
        assert record.live is True
        assert record.asset.entitlements == {"public", "team-platform"}
        assert [s.asset_id for s in record.asset.sources] == [
            "asset-3fa85f64d1a2b3c4",
            "asset-0123456789abcdef",
        ]
        assert record.scene.index == 0 and record.scene.end_seconds == 13.347
        assert len(record.scene.keyframes) == 2 and record.scene.audio_clip is not None
        (transcript,) = record.enrichments
        assert transcript.pin == models.EnrichmentPin("transcript", "e41a9c27b0d5")
        assert [s.text for s in transcript.segments][0].startswith("Welcome back.")

    def test_when_decoded_then_version_recomputes_from_its_parts(self) -> None:
        """The rule scene-api publishes: hash(set version + index + sorted enrichment versions)."""
        record = scene_record_from_json(load("scene-record"))
        lines = [
            record.scene_set_version,
            str(record.scene.index),
            *sorted(e.version for e in record.enrichments),
        ]

        assert record.version == "sha256:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()

    def test_when_encoded_again_then_identical_to_sample(self) -> None:
        sample = load("scene-record")

        assert encode_scene_record(scene_record_from_json(sample)) == sample

    def test_when_schema_version_differs_then_refused(self) -> None:
        sample = load("scene-record")
        sample["schema_version"] = 2

        with pytest.raises(SceneApiError, match="schema_version"):
            scene_record_from_json(sample)


class TestGivenSummarySample:
    """The published scene set summary."""

    def test_when_decoded_then_coverage_kept(self) -> None:
        sample = load("scene-set-summary")

        summary = summary_from_json(sample)

        assert summary.scene_count == 2 and summary.live is True
        assert summary.created_at.isoformat() == "2026-10-27T10:00:00+00:00"
        assert [(c.enricher, c.complete) for c in summary.enrichments] == [
            ("description", True),
            ("transcript", False),
        ]
        assert encode_summary(summary) == sample


class TestGivenAssetVersionSample:
    """The published asset version."""

    def test_when_decoded_then_liveness_kept(self) -> None:
        sample = load("asset-version")

        assert asset_version_from_json(sample) == models.AssetVersionView(sample["version"], True)
