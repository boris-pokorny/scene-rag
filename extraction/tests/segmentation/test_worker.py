"""The composed worker end to end: real files, MongoDB, PySceneDetect and ffmpeg (`integration`)."""

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
from pymongo import MongoClient

from scene_rag.extraction.ingestion import bootstrap as ingestion
from scene_rag.extraction.ingestion import settings as ingestion_settings
from scene_rag.extraction.segmentation import bootstrap, settings
from scene_rag.extraction.segmentation.scene_store import COLLECTION, Document, from_document

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
    """Both steps configured over ``tmp_path``; returns the media directory."""
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv(ingestion_settings.REGISTRY_URI_ENV, mongo_uri)
    monkeypatch.setenv(ingestion_settings.REGISTRY_DATABASE_ENV, database)
    monkeypatch.setenv(ingestion_settings.SOURCE_ROOT_ENV, str(media))
    monkeypatch.setenv(ingestion_settings.SOURCE_ID_ENV, "local")
    monkeypatch.setenv(ingestion_settings.SOURCE_ENTITLEMENTS_ENV, "public")
    monkeypatch.setenv(settings.SCENE_STORE_URI_ENV, mongo_uri)
    monkeypatch.setenv(settings.SCENE_STORE_DATABASE_ENV, database)
    monkeypatch.setenv(settings.DERIVATIVE_ROOT_ENV, str(tmp_path / "derivatives"))
    monkeypatch.delenv(settings.FFMPEG_BINARY_ENV, raising=False)
    return media


def synthesize(path: Path, tone_hz: int = 440) -> None:
    """Write a four-second clip with a tone and a hard cut from black to white at two seconds."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("no ffmpeg on PATH")
    subprocess.run(
        [ffmpeg, "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=160x120:r=25:d=2",
         "-f", "lavfi", "-i", "color=c=white:s=160x120:r=25:d=2",
         "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:duration=4",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a", "-c:v", "mpeg4", "-q:v", "2", "-c:a", "aac", "-shortest",
         str(path)],
        check=True,
    )  # fmt: skip
    aged = time.time() - 60
    os.utime(path, (aged, aged))


class TestGivenIngestedVideo:
    """A synthesized two-scene clip registered by ingestion."""

    def test_when_segmented_then_scene_set_and_derivatives_written_once(
        self, environment: Path, mongo_client: MongoClient[Document], database: str, tmp_path: Path
    ) -> None:
        synthesize(environment / "clip.mp4")
        ingestion.run_worker(once=True)

        bootstrap.run_worker(once=True)
        bootstrap.run_worker(once=True)

        documents = list(mongo_client[database][COLLECTION].find())
        assert len(documents) == 1
        scene_set = from_document(documents[0])
        (asset,) = mongo_client[database]["asset_versions"].find()
        assert scene_set.asset_version == asset["_id"]
        assert scene_set.lineage.inputs == (f"asset-version:{asset['_id']}",)
        assert len(scene_set.scenes) == 2
        assert scene_set.scenes[1].start_seconds == pytest.approx(2.0, abs=0.1)
        for scene in scene_set.scenes:
            assert scene.keyframes and scene.audio_clip is not None
            for keyframe in scene.keyframes:
                assert Path(keyframe.derivative.uri.removeprefix("file://")).is_file()
        derivatives = sorted(p.name for p in (tmp_path / "derivatives").rglob("*") if p.is_file())
        assert len(derivatives) == len(
            {
                d.digest
                for s in scene_set.scenes
                for d in (*[k.derivative for k in s.keyframes], s.audio_clip)
                if d
            }
        )


class TestGivenTwoIngestedVideos:
    """Two synthesized clips registered by ingestion, segmented serially and then in parallel."""

    def test_when_segmented_in_parallel_then_same_scene_sets_and_derivatives(
        self,
        environment: Path,
        mongo_client: MongoClient[Document],
        database: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        synthesize(environment / "first.mp4")
        synthesize(environment / "second.mp4", tone_hz=880)
        ingestion.run_worker(once=True)
        bootstrap.run_worker(once=True)
        serial = masked(mongo_client[database][COLLECTION].find())
        files = sorted(p for p in (tmp_path / "derivatives").rglob("*") if p.is_file())
        assert len(serial) == 2 and files

        parallel_database = f"{database}_parallel"
        monkeypatch.setenv(settings.SCENE_STORE_DATABASE_ENV, parallel_database)
        monkeypatch.setenv(settings.WORKERS_ENV, "2")
        monkeypatch.setenv(settings.AUDIO_CLIP_WORKERS_ENV, "2")
        try:
            bootstrap.run_worker(once=True)
            parallel = masked(mongo_client[parallel_database][COLLECTION].find())
        finally:
            mongo_client.drop_database(parallel_database)

        assert parallel == serial
        assert sorted(p for p in (tmp_path / "derivatives").rglob("*") if p.is_file()) == files


def masked(documents: Iterable[Document]) -> list[Document]:
    """Scene set documents without the write timestamp, in asset version order."""
    masked = []
    for document in documents:
        document.pop("created_at")
        masked.append(document)
    return sorted(masked, key=lambda d: str(d["asset_version"]))
