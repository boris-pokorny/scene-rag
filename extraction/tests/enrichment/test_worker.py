"""The composed worker end to end: MongoDB, real derivatives, fake models (`integration`).

The vision and speech models are replaced by fakes through `bootstrap.create_enrichers`, so the
case needs MongoDB, ffmpeg and the segmentation extra but no model server; the model-backed
backends have their own opt-in cases.
"""

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from pymongo import MongoClient

from scene_rag.extraction.enrichment import bootstrap, settings
from scene_rag.extraction.enrichment.config import EnrichmentConfig
from scene_rag.extraction.enrichment.derivative_reader import FileDerivativeReader
from scene_rag.extraction.enrichment.description import DescriptionEnricher
from scene_rag.extraction.enrichment.enrichment_store import COLLECTION, Document, from_document
from scene_rag.extraction.enrichment.transcription import TranscriptionEnricher
from scene_rag.extraction.ingestion import bootstrap as ingestion
from scene_rag.extraction.ingestion import settings as ingestion_settings
from tests.support.enrichment_fakes import FakeTranscriber, FakeVisionModel

pytestmark = pytest.mark.integration


@pytest.fixture
def database(mongo_client: MongoClient[Document]) -> Iterator[str]:
    name = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield name
    mongo_client.drop_database(name)


@pytest.fixture
def environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database: str, mongo_uri: str
) -> Path:
    """All three steps configured over ``tmp_path``; returns the media directory."""
    pytest.importorskip("scenedetect")
    from scene_rag.extraction.segmentation import settings as segmentation_settings

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv(ingestion_settings.REGISTRY_URI_ENV, mongo_uri)
    monkeypatch.setenv(ingestion_settings.REGISTRY_DATABASE_ENV, database)
    monkeypatch.setenv(ingestion_settings.SOURCE_ROOT_ENV, str(media))
    monkeypatch.setenv(ingestion_settings.SOURCE_ID_ENV, "local")
    monkeypatch.setenv(ingestion_settings.SOURCE_ENTITLEMENTS_ENV, "public")
    monkeypatch.setenv(segmentation_settings.SCENE_STORE_URI_ENV, mongo_uri)
    monkeypatch.setenv(segmentation_settings.SCENE_STORE_DATABASE_ENV, database)
    monkeypatch.setenv(segmentation_settings.DERIVATIVE_ROOT_ENV, str(tmp_path / "derivatives"))
    monkeypatch.delenv(segmentation_settings.FFMPEG_BINARY_ENV, raising=False)
    monkeypatch.setenv(settings.ENRICHMENT_STORE_URI_ENV, mongo_uri)
    monkeypatch.setenv(settings.ENRICHMENT_STORE_DATABASE_ENV, database)
    return media


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeVisionModel, FakeTranscriber]:
    """Replace the model-backed enrichers with the reference enrichers over fake models."""
    vision = FakeVisionModel("People talk in front of a screen.")
    transcriber = FakeTranscriber()

    def create_enrichers(loaded: object) -> tuple[DescriptionEnricher, TranscriptionEnricher]:
        config = EnrichmentConfig()
        reader = FileDerivativeReader()
        return (
            DescriptionEnricher(vision=vision, derivatives=reader, config=config.description),
            TranscriptionEnricher(
                transcriber=transcriber, derivatives=reader, config=config.transcription
            ),
        )

    monkeypatch.setattr(bootstrap, "create_enrichers", create_enrichers)
    return vision, transcriber


def synthesize(path: Path) -> None:
    """Write a four-second clip with a tone and a hard cut from black to white at two seconds."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("no ffmpeg on PATH")
    subprocess.run(
        [ffmpeg, "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=160x120:r=25:d=2",
         "-f", "lavfi", "-i", "color=c=white:s=160x120:r=25:d=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a", "-c:v", "mpeg4", "-q:v", "2", "-c:a", "aac", "-shortest",
         str(path)],
        check=True,
    )  # fmt: skip
    aged = time.time() - 60
    os.utime(path, (aged, aged))


class TestGivenSegmentedVideo:
    """A synthesized two-scene clip, ingested and segmented."""

    def test_when_enriched_then_one_record_per_scene_and_enricher_once(
        self,
        environment: Path,
        fake_models: tuple[FakeVisionModel, FakeTranscriber],
        mongo_client: MongoClient[Document],
        database: str,
    ) -> None:
        from scene_rag.extraction.segmentation import bootstrap as segmentation

        synthesize(environment / "clip.mp4")
        ingestion.run_worker(once=True)
        segmentation.run_worker(once=True)
        vision, transcriber = fake_models

        bootstrap.run_worker(once=True)
        bootstrap.run_worker(once=True)

        (scene_set,) = mongo_client[database]["scene_sets"].find()
        records = [from_document(d) for d in mongo_client[database][COLLECTION].find()]
        assert {(r.enricher, r.scene_index) for r in records} == {
            (name, index)
            for name in ("description", "transcript")
            for index in range(len(scene_set["scenes"]))
        }
        assert len(records) == 2 * len(scene_set["scenes"]) == 4
        assert len(vision.calls) == 2 and len(transcriber.calls) == 2
        for record in records:
            assert record.scene_set_version == scene_set["_id"]
            assert record.asset_version == scene_set["asset_version"]
            assert record.lineage.inputs[0] == f"scene-set:{scene_set['_id']}"
            assert len(record.lineage.inputs) > 1
            scene = scene_set["scenes"][record.scene_index]
            for segment in record.segments:
                assert scene["start_seconds"] <= segment.start_seconds <= segment.end_seconds
                assert segment.end_seconds <= scene["end_seconds"] + 1e-6
        descriptions = [r for r in records if r.enricher == "description"]
        assert all(r.segments[0].text == "People talk in front of a screen." for r in descriptions)
