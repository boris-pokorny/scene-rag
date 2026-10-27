"""The composed worker against real files and a real MongoDB (`integration`)."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from pymongo import MongoClient

from scene_rag.extraction.ingestion import bootstrap, settings
from scene_rag.extraction.ingestion.registry import COLLECTION, Document, from_document

pytestmark = pytest.mark.integration


@pytest.fixture
def database(mongo_client: MongoClient[Document]) -> Iterator[str]:
    name = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield name
    mongo_client.drop_database(name)


@pytest.fixture
def media_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database: str, mongo_uri: str
) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    monkeypatch.setenv(settings.REGISTRY_URI_ENV, mongo_uri)
    monkeypatch.setenv(settings.REGISTRY_DATABASE_ENV, database)
    monkeypatch.setenv(settings.SOURCE_ROOT_ENV, str(root))
    monkeypatch.setenv(settings.SOURCE_ID_ENV, "local")
    monkeypatch.setenv(settings.SOURCE_ENTITLEMENTS_ENV, "public")
    monkeypatch.setenv(settings.POLL_INTERVAL_SECONDS_ENV, "1")
    return root


def live_filenames(client: MongoClient[Document], database: str) -> dict[str, list[str]]:
    collection = client[database][COLLECTION]
    return {
        record.version[:20]: sorted(reference.filename for reference in record.sources)
        for record in map(from_document, collection.find({"retired_at": None}))
    }


class TestGivenMediaDirectory:
    """Real files and a real MongoDB through `run_worker(once=True)`."""

    def test_when_files_change_then_registry_follows(
        self, media_dir: Path, mongo_client: MongoClient[Document], database: str
    ) -> None:
        (media_dir / "a.mp4").write_bytes(b"first")
        bootstrap.run_worker(once=True)
        assert live_filenames(mongo_client, database) == {"sha256:a7937b64b8caa": ["a.mp4"]}

        (media_dir / "copy.mp4").write_bytes(b"first")
        bootstrap.run_worker(once=True)
        assert live_filenames(mongo_client, database) == {
            "sha256:a7937b64b8caa": ["a.mp4", "copy.mp4"]
        }

        (media_dir / "copy.mp4").write_bytes(b"second")
        bootstrap.run_worker(once=True)
        assert live_filenames(mongo_client, database) == {
            "sha256:a7937b64b8caa": ["a.mp4"],
            "sha256:16367aacb67a4": ["copy.mp4"],
        }

        (media_dir / "copy.mp4").unlink()
        bootstrap.run_worker(once=True)
        assert live_filenames(mongo_client, database) == {"sha256:a7937b64b8caa": ["a.mp4"]}
        tombstone = mongo_client[database][COLLECTION].find_one({"retired_at": {"$ne": None}})
        assert tombstone is not None and tombstone["sources"] == []

    def test_when_registry_unreachable_then_runtime_error(
        self, media_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(settings.REGISTRY_URI_ENV, "mongodb://127.0.0.1:1")

        with pytest.raises(RuntimeError, match="Could not reach MongoDB"):
            bootstrap.run_worker(once=True)
