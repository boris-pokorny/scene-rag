"""The composed worker end to end: a stub scene-api over HTTP and MongoDB (`integration`).

The stub serves the published documents the fakes' encoder renders, so the case needs no
extraction package at all and runs from a sparse checkout of this service.
"""

import json
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from pymongo import MongoClient

from scene_rag.retrieval.chunking import bootstrap, models, settings
from scene_rag.retrieval.chunking.chunk_store import (
    CHUNK_SETS_COLLECTION,
    CHUNKS_COLLECTION,
    Document,
    chunk_from_document,
    chunk_set_from_document,
)
from tests.support.chunking_fakes import (
    encode_asset_version,
    encode_page,
    encode_scene_record,
    encode_summary,
    make_records,
    sha256_version,
    summary_of,
)

pytestmark = pytest.mark.integration

FIRST = make_records(sha256_version(b"first"))
SECOND = make_records(sha256_version(b"second"), bounds=((0.0, 2.0),))
SCENE_SETS = {summary_of(r).version: r for r in (FIRST, SECOND)}


class StubSceneApi(BaseHTTPRequestHandler):
    """The routes chunking uses, paged one item at a time to exercise the cursors."""

    def _answer(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - the server's naming
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        after = query.get("after", [None])[0]
        if parts.path == "/health":
            self._answer(200, {"status": "ok"})
        elif parts.path == "/scene-sets":
            versions = sorted(SCENE_SETS)
            remaining = [v for v in versions if after is None or v > after]
            items = [encode_summary(summary_of(SCENE_SETS[v])) for v in remaining[:1]]
            self._answer(200, encode_page(items, remaining[0] if len(remaining) > 1 else None))
        elif parts.path.startswith("/scene-sets/") and parts.path.endswith("/scenes"):
            version = parts.path[len("/scene-sets/") : -len("/scenes")]
            records = SCENE_SETS[version]
            start = 0 if after is None else int(after) + 1
            page = records[start : start + 1]
            more = start + 1 < len(records)
            self._answer(
                200,
                encode_page(
                    [encode_scene_record(r) for r in page],
                    str(page[-1].scene.index) if more else None,
                ),
            )
        elif parts.path.startswith("/asset-versions/"):
            version = parts.path[len("/asset-versions/") :]
            self._answer(200, encode_asset_version(models.AssetVersionView(version, True)))
        else:
            self._answer(404, {"error": {"code": "not_found", "message": self.path}})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the server's naming
        return


@pytest.fixture
def scene_api_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubSceneApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def database(mongo_client: MongoClient[Document]) -> Iterator[str]:
    name = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    yield name
    mongo_client.drop_database(name)


@pytest.fixture
def environment(
    monkeypatch: pytest.MonkeyPatch, database: str, mongo_uri: str, scene_api_url: str
) -> None:
    monkeypatch.setenv(settings.SCENE_API_URL_ENV, scene_api_url)
    monkeypatch.setenv(settings.CHUNK_STORE_URI_ENV, mongo_uri)
    monkeypatch.setenv(settings.CHUNK_STORE_DATABASE_ENV, database)
    monkeypatch.delenv(settings.POLL_INTERVAL_SECONDS_ENV, raising=False)


class TestGivenPublishedSceneSets:
    """Two complete scene sets served by the stub."""

    def test_when_worker_runs_twice_then_one_chunk_per_scene_and_one_chunk_set_once(
        self, environment: None, mongo_client: MongoClient[Document], database: str
    ) -> None:
        bootstrap.run_worker(once=True)
        bootstrap.run_worker(once=True)

        chunks = [chunk_from_document(d) for d in mongo_client[database][CHUNKS_COLLECTION].find()]
        chunk_sets = [
            chunk_set_from_document(d) for d in mongo_client[database][CHUNK_SETS_COLLECTION].find()
        ]
        assert {(c.scene_set_version, c.chunk_index) for c in chunks} == {
            (summary_of(r).version, i) for r in (FIRST, SECOND) for i in range(len(r))
        }
        assert {(s.scene_set_version, s.chunk_count) for s in chunk_sets} == {
            (summary_of(FIRST).version, 3),
            (summary_of(SECOND).version, 1),
        }
        for chunk in chunks:
            record = SCENE_SETS[chunk.scene_set_version][chunk.chunk_index]
            assert chunk.lineage.inputs == (f"scene-record:{record.version}",)
            assert chunk.entitlements == {"public"}
            assert "description of scene" in chunk.text and "transcript of scene" in chunk.text
