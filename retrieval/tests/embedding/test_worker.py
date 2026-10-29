"""The composed worker end to end: a stub Ollama over HTTP, MongoDB and Qdrant (`integration`).

The Chunk Store is seeded the way the chunking step writes it, through its own encoder, so
the case exercises the read model against real documents.
"""

import json
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pymongo import MongoClient

from scene_rag.retrieval.chunking.chunk_store import (
    CHUNK_SETS_COLLECTION,
    CHUNKS_COLLECTION,
    Document,
    MongoChunkStore,
    chunk_set_to_document,
    chunk_to_document,
)
from scene_rag.retrieval.embedding import bootstrap, models, settings
from scene_rag.retrieval.embedding.index_store import INDEX_SETS_COLLECTION, index_set_from_document
from scene_rag.retrieval.embedding.qdrant_index import QdrantSearchIndex
from tests.support import chunking_fakes

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]

FIRST = chunking_fakes.make_records(chunking_fakes.sha256_version(b"first"))
SECOND = chunking_fakes.make_records(chunking_fakes.sha256_version(b"second"), bounds=((0.0, 2.0),))
CHUNKING_FP = "c7a1e2f3b4d5"
DIMENSIONS = 6


class StubOllama(BaseHTTPRequestHandler):
    """The routes embedding uses: version, tags and embed with fixed-length vectors."""

    def _answer(self, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - the server's naming
        if self.path == "/api/version":
            self._answer({"version": "stub"})
        else:
            self._answer(
                {
                    "models": [
                        {
                            "name": "embeddinggemma:latest",
                            "model": "embeddinggemma:latest",
                            "digest": "ab" * 32,
                        }
                    ]
                }
            )

    def do_POST(self) -> None:  # noqa: N802 - the server's naming
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._answer(
            {"embeddings": [[float(len(text) % 7)] * DIMENSIONS for text in payload["input"]]}
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the server's naming
        return


@pytest.fixture
def ollama_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def database(mongo_client: MongoClient[Document]) -> Iterator[str]:
    name = f"scene_rag_test_{uuid.uuid4().hex[:8]}"
    MongoChunkStore(mongo_client, name)
    mongo_client[name][CHUNKS_COLLECTION].insert_many(
        [
            chunk_to_document(chunking_fakes.make_chunk(records, i, CHUNKING_FP))
            for records in (FIRST, SECOND)
            for i in range(len(records))
        ]
    )
    mongo_client[name][CHUNK_SETS_COLLECTION].insert_many(
        [
            chunk_set_to_document(chunking_fakes.make_chunk_set(r, CHUNKING_FP))
            for r in (FIRST, SECOND)
        ]
    )
    yield name
    mongo_client.drop_database(name)


@pytest.fixture
def environment(
    monkeypatch: pytest.MonkeyPatch,
    database: str,
    mongo_uri: str,
    qdrant_url: str,
    ollama_url: str,
) -> None:
    monkeypatch.setenv(settings.CHUNK_STORE_URI_ENV, mongo_uri)
    monkeypatch.setenv(settings.CHUNK_STORE_DATABASE_ENV, database)
    monkeypatch.setenv(settings.SEARCH_INDEX_URL_ENV, qdrant_url)
    monkeypatch.setenv(settings.OLLAMA_URL_ENV, ollama_url)
    monkeypatch.setenv(settings.BATCH_SIZE_ENV, "2")
    monkeypatch.delenv(settings.POLL_INTERVAL_SECONDS_ENV, raising=False)


class TestGivenChunkedSceneSets:
    """Two chunk sets in the Chunk Store, the stub model behind Ollama's API."""

    def test_when_worker_runs_twice_then_one_point_per_chunk_and_one_index_set_once(
        self,
        environment: None,
        mongo_client: MongoClient[Document],
        database: str,
        qdrant_index: QdrantSearchIndex,
    ) -> None:
        collection: str | None = None
        try:
            bootstrap.run_worker(once=True)
            bootstrap.run_worker(once=True)

            index_sets = [
                index_set_from_document(d)
                for d in mongo_client[database][INDEX_SETS_COLLECTION].find()
            ]
            assert {(s.scene_set_version, s.vector_count) for s in index_sets} == {
                (FIRST[0].scene_set_version, 3),
                (SECOND[0].scene_set_version, 1),
            }
            (collection,) = {s.collection for s in index_sets}
            assert qdrant_index.count(collection) == 4
            for index_set in index_sets:
                assert (
                    qdrant_index.count(collection, chunk_set_version=index_set.chunk_set_version)
                    == index_set.vector_count
                )
            chunk = chunking_fakes.make_chunk(FIRST, 1, CHUNKING_FP)
            (entry,) = qdrant_index.get_entries(collection, [models.point_id(chunk.version)])
            assert entry.text == chunk.text and entry.entitlements == {"public"}
            assert entry.lineage.inputs == (f"chunk:{chunk.version}",)
        finally:
            if collection is not None:
                qdrant_index.drop_collection(collection)
