"""The composition root: settings, connections, index selection and checks, the entry points."""

import logging
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, cast

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.server.mcpserver import MCPServer
from pymongo.errors import ServerSelectionTimeoutError

from scene_rag.retrieval.query_api import bootstrap, settings
from scene_rag.retrieval.query_api.caller import current_caller
from scene_rag.retrieval.query_api.config import (
    GeneratorConfig,
    QueryConfig,
    UpstreamConfig,
)
from scene_rag.retrieval.query_api.fingerprint import (
    FINGERPRINTED_MODULES,
    compute_query_config_id,
)
from scene_rag.retrieval.query_api.models import Caller
from scene_rag.retrieval.query_api.protocols import IndexUnavailableError
from tests.support.query_fakes import (
    EMBEDDING_MODEL,
    GENERATOR_MODEL,
    T0,
    FakeAnswerGenerator,
    FakeIndexLedger,
    FakeQueryEmbedder,
    FakeSceneReader,
    FakeSearchReader,
    FakeTransport,
    index_document,
    make_index_view,
)

REQUIRED = {
    settings.CHUNK_STORE_URI_ENV: "mongodb://localhost:27017",
    settings.CHUNK_STORE_DATABASE_ENV: "scene_rag_retrieval",
    settings.SEARCH_INDEX_URL_ENV: "http://localhost:6333",
    settings.SCENE_API_URL_ENV: "http://127.0.0.1:8080",
}
INDEX = make_index_view()
TAGS: dict[str, Any] = {
    "models": [
        {"name": "embeddinggemma:latest", "model": "embeddinggemma:latest", "digest": "ab" * 32},
        {"name": "qwen2.5:3b", "model": "qwen2.5:3b", "digest": "cd" * 32},
    ]
}
QDRANT_COLLECTION: dict[str, Any] = {
    "result": {
        "points_count": 7,
        "config": {"params": {"vectors": {"size": 8, "distance": "Cosine"}}},
    }
}


def transport(**overrides: Any) -> FakeTransport:  # noqa: ANN401 - scripted answers of any shape
    answers: dict[str, Any] = {
        "GET /api/version": {"version": "0.40.12"},
        "GET /api/tags": TAGS,
        "POST /api/embed": {"embeddings": [[0.1] * 8]},
        "POST /api/pull": {"status": "success"},
        "GET /": {"version": "1.20.0"},
        f"GET /collections/{INDEX.collection}": QDRANT_COLLECTION,
        "GET /health": {"status": "ok"},
    }
    answers.update(overrides)
    return FakeTransport(answers)


class TestGivenEnvironment:
    """Variables as a deployment sets them."""

    def test_when_complete_then_settings_bound_with_defaults(self) -> None:
        loaded = bootstrap.load_settings(REQUIRED)

        assert loaded.chunk_store_uri == "mongodb://localhost:27017"
        assert loaded.scene_api_url == "http://127.0.0.1:8080"
        assert (loaded.host, loaded.port) == ("127.0.0.1", 8081)
        assert loaded.allowed_hosts == ("127.0.0.1", "localhost")
        assert loaded.ollama.url == "http://localhost:11434"
        assert loaded.ollama.generator_timeout_seconds == 300.0

    def test_when_optional_given_then_bound(self) -> None:
        loaded = bootstrap.load_settings(
            {
                **REQUIRED,
                settings.HOST_ENV: "0.0.0.0",
                settings.PORT_ENV: "9000",
                settings.OLLAMA_URL_ENV: "http://ollama:11434",
                settings.OLLAMA_TIMEOUT_SECONDS_ENV: "10",
                settings.GENERATOR_TIMEOUT_SECONDS_ENV: "600",
                settings.SEARCH_INDEX_TIMEOUT_SECONDS_ENV: "2.5",
                settings.SCENE_API_TIMEOUT_SECONDS_ENV: "4",
                settings.ALLOWED_HOSTS_ENV: " query-api, localhost ,",
            }
        )

        assert (loaded.host, loaded.port) == ("0.0.0.0", 9000)
        assert loaded.ollama.url == "http://ollama:11434"
        assert (loaded.ollama.timeout_seconds, loaded.ollama.generator_timeout_seconds) == (10, 600)
        assert (loaded.search_index_timeout_seconds, loaded.scene_api_timeout_seconds) == (2.5, 4)
        assert loaded.allowed_hosts == ("query-api", "localhost")

    @pytest.mark.parametrize("name", sorted(REQUIRED))
    def test_when_required_missing_then_named(self, name: str) -> None:
        environ = {k: v for k, v in REQUIRED.items() if k != name}

        with pytest.raises(RuntimeError, match=name):
            bootstrap.load_settings(environ)

    @pytest.mark.parametrize(
        "name, value",
        [
            (settings.PORT_ENV, "eighty"),
            (settings.PORT_ENV, "-1"),
            (settings.OLLAMA_TIMEOUT_SECONDS_ENV, "soon"),
            (settings.GENERATOR_TIMEOUT_SECONDS_ENV, "0"),
            (settings.SEARCH_INDEX_URL_ENV, "qdrant:6333"),
            (settings.ALLOWED_HOSTS_ENV, " , "),
        ],
    )
    def test_when_value_invalid_then_runtime_error_names_it(self, name: str, value: str) -> None:
        with pytest.raises(RuntimeError, match="invalid settings"):
            bootstrap.load_settings({**REQUIRED, name: value})


class TestGivenBackends:
    """Connection checks over scripted transports and a fake Mongo client factory."""

    def test_when_mongo_answers_then_client_returned(self) -> None:
        calls: list[tuple[str, Mapping[str, object]]] = []

        class Client:
            def __init__(self, uri: str, **options: object) -> None:
                calls.append((uri, options))
                self.admin = self

            def command(self, name: str) -> None:
                return None

        client = bootstrap.connect_mongo(
            "mongodb://u:p@h", redacted="mongodb://***@h", client_factory=factory(Client)
        )

        assert isinstance(client, Client)
        assert calls == [("mongodb://u:p@h", {"tz_aware": True, "serverSelectionTimeoutMS": 2000})]

    def test_when_mongo_silent_then_runtime_error_redacted(self) -> None:
        class Client:
            def __init__(self, uri: str, **options: object) -> None:
                self.admin = self

            def command(self, name: str) -> None:
                raise ServerSelectionTimeoutError("no route")

        with pytest.raises(RuntimeError, match="mongodb://\\*\\*\\*@h") as exc_info:
            bootstrap.connect_mongo(
                "mongodb://u:p@h", redacted="mongodb://***@h", client_factory=factory(Client)
            )
        assert "u:p" not in str(exc_info.value)

    def test_when_ollama_answers_then_client_with_both_timeouts(self) -> None:
        ollama = bootstrap.connect_ollama(
            settings.OllamaSettings("http://ollama:11434", 10.0, 600.0), transport()
        )

        assert ollama.base_url == "http://ollama:11434"

    def test_when_ollama_silent_then_runtime_error_with_hint(self) -> None:
        fake = transport()
        fake.unreachable("GET /api/version")

        with pytest.raises(RuntimeError, match=settings.OLLAMA_URL_ENV):
            bootstrap.connect_ollama(settings.OllamaSettings(), fake)

    def test_when_qdrant_answers_then_reader(self) -> None:
        reader = bootstrap.connect_search_index(bootstrap.load_settings(REQUIRED), transport())

        assert reader.version() == "1.20.0"

    def test_when_qdrant_silent_then_runtime_error_with_hint(self) -> None:
        fake = transport()
        fake.unreachable("GET /")

        with pytest.raises(RuntimeError, match=settings.SEARCH_INDEX_URL_ENV):
            bootstrap.connect_search_index(bootstrap.load_settings(REQUIRED), fake)

    def test_when_scene_api_answers_then_reader(self) -> None:
        reader = bootstrap.connect_scene_api(bootstrap.load_settings(REQUIRED), transport())

        reader.health()

    def test_when_scene_api_silent_then_runtime_error_with_hint(self) -> None:
        fake = transport()
        fake.unreachable("GET /health")

        with pytest.raises(RuntimeError, match=settings.SCENE_API_URL_ENV):
            bootstrap.connect_scene_api(bootstrap.load_settings(REQUIRED), fake)


class TestGivenLedger:
    """Index selection over a fake ledger."""

    OLD = make_index_view("aaaaaaaaaaaa", created_at=T0 - timedelta(days=1))

    def test_when_pinned_and_present_then_that_index(self) -> None:
        config = QueryConfig(upstream=UpstreamConfig(embedding_fingerprint="aaaaaaaaaaaa"))

        assert bootstrap.select_index(FakeIndexLedger(self.OLD, INDEX), config) == self.OLD

    def test_when_pinned_and_absent_then_runtime_error(self) -> None:
        config = QueryConfig(upstream=UpstreamConfig(embedding_fingerprint="cccccccccccc"))

        with pytest.raises(RuntimeError, match="cccccccccccc"):
            bootstrap.select_index(FakeIndexLedger(self.OLD, INDEX), config)

    def test_when_unpinned_then_newest(self) -> None:
        assert bootstrap.select_index(FakeIndexLedger(self.OLD, INDEX), QueryConfig()) == INDEX

    def test_when_ledger_empty_then_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="empty"):
            bootstrap.select_index(FakeIndexLedger(), QueryConfig())

    def test_when_ledger_fails_then_runtime_error(self) -> None:
        ledger = FakeIndexLedger()
        ledger.failure = IndexUnavailableError("down")

        with pytest.raises(RuntimeError, match="cannot be read"):
            bootstrap.select_index(ledger, QueryConfig())


class TestGivenIndex:
    """Checks that the index can be served."""

    def test_when_everything_matches_then_embedder_for_the_index(self) -> None:
        fake = transport()
        ollama = bootstrap.connect_ollama(settings.OllamaSettings(), fake)
        search_index = bootstrap.connect_search_index(bootstrap.load_settings(REQUIRED), fake)

        embedder = bootstrap.check_index(ollama, search_index, INDEX)

        assert embedder.model == EMBEDDING_MODEL
        assert embedder.dimensions == 8
        assert fake.calls[-1][2] == {
            "model": "embeddinggemma",
            "input": ["task: search result | query: dimension probe"],
            "truncate": True,
            "options": {"num_ctx": 2048},
        }

    def test_when_model_missing_then_runtime_error_with_pull_hint(self) -> None:
        fake = transport(**{"GET /api/tags": {"models": []}})
        ollama = bootstrap.connect_ollama(settings.OllamaSettings(), fake)
        search_index = bootstrap.connect_search_index(bootstrap.load_settings(REQUIRED), fake)

        with pytest.raises(RuntimeError, match="pull-models"):
            bootstrap.check_index(ollama, search_index, INDEX)

    def test_when_digest_differs_then_runtime_error_naming_both(self) -> None:
        fake = transport()
        ollama = bootstrap.connect_ollama(settings.OllamaSettings(), fake)
        search_index = bootstrap.connect_search_index(bootstrap.load_settings(REQUIRED), fake)
        other = make_index_view(
            model=EMBEDDING_MODEL.__class__("embeddinggemma", "sha256:" + "ef" * 32)
        )

        with pytest.raises(RuntimeError, match="ab" * 32) as exc_info:
            bootstrap.check_index(ollama, search_index, other)
        assert "ef" * 32 in str(exc_info.value)

    @pytest.mark.parametrize(
        "collection, message",
        [
            (None, "does not exist"),
            (
                {"result": {"config": {"params": {"vectors": {"size": 4, "distance": "Cosine"}}}}},
                "4-dimensional",
            ),
            ({"result": {"config": {}}}, "Could not inspect"),
        ],
    )
    def test_when_collection_unusable_then_runtime_error(
        self, collection: dict[str, Any] | None, message: str
    ) -> None:
        fake = transport()
        key = f"GET /collections/{INDEX.collection}"
        if collection is None:
            del fake.answers[key]
        else:
            fake.answers[key] = collection
        ollama = bootstrap.connect_ollama(settings.OllamaSettings(), fake)
        search_index = bootstrap.connect_search_index(bootstrap.load_settings(REQUIRED), fake)

        with pytest.raises(RuntimeError, match=message):
            bootstrap.check_index(ollama, search_index, INDEX)

    def test_when_probe_embeds_wrongly_then_runtime_error(self) -> None:
        fake = transport(**{"POST /api/embed": {"embeddings": [[0.1] * 3]}})
        ollama = bootstrap.connect_ollama(settings.OllamaSettings(), fake)
        search_index = bootstrap.connect_search_index(bootstrap.load_settings(REQUIRED), fake)

        with pytest.raises(RuntimeError, match="probe"):
            bootstrap.check_index(ollama, search_index, INDEX)


class TestGivenGenerator:
    """The generator resolved on Ollama, and the service assembled over everything."""

    def test_when_model_present_then_generator_with_digest(self) -> None:
        ollama = bootstrap.connect_ollama(settings.OllamaSettings(), transport())

        assert bootstrap.create_generator(ollama, QueryConfig()).model == GENERATOR_MODEL

    def test_when_model_missing_then_runtime_error_with_pull_hint(self) -> None:
        ollama = bootstrap.connect_ollama(settings.OllamaSettings(), transport())

        with pytest.raises(RuntimeError, match="pull-models"):
            bootstrap.create_generator(ollama, QueryConfig(generator=GeneratorConfig(name="x")))

    def test_when_service_created_then_served_names_config_and_index(self) -> None:
        config = QueryConfig()

        service = bootstrap.create_service(
            embedder=FakeQueryEmbedder(INDEX),
            search_index=FakeSearchReader(),
            generator=FakeAnswerGenerator(),
            scenes=FakeSceneReader(),
            index=INDEX,
            config=config,
        )

        assert service.served.query_config == compute_query_config_id(
            config, FINGERPRINTED_MODULES, GENERATOR_MODEL.digest
        )
        assert service.served.embedding_fingerprint == INDEX.fingerprint
        assert service.served.collection == INDEX.collection
        assert bootstrap.NoRetiredFilter().is_retired("sha256:aa") is False


class TestGivenEntryPoints:
    """`run_server` over fakes for every backend, and `pull_models`."""

    @pytest.fixture
    def wired(self, monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
        fake = transport()
        for name, value in REQUIRED.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(bootstrap, "UrllibTransport", lambda: fake)

        class Client:
            def __init__(self, uri: str, **options: object) -> None:
                self.admin = self

            def command(self, name: str) -> None:
                return None

            def __getitem__(self, database: str) -> dict[str, Any]:
                return {"indexes": _Collection(index_document(INDEX))}

            def __enter__(self) -> Client:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(
            bootstrap, "connect_mongo", lambda uri, *, redacted: factory(Client)(uri)
        )
        return fake

    def test_when_http_then_app_served_on_host_and_port(
        self,
        wired: FakeTransport,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(settings.PORT_ENV, "0")
        served: list[tuple[str, int, list[str]]] = []
        statuses: list[int] = []

        def serve(app: FastAPI, host: str, port: int) -> None:
            served.append((host, port, [getattr(r, "path", "") for r in app.routes]))
            client = TestClient(app)
            statuses.extend(
                client.get("/health", headers={"Host": name}).status_code
                for name in ("localhost:8081", "rebound.example")
            )

        with caplog.at_level(logging.INFO):
            bootstrap.run_server(serve=serve)

        (host, port, paths) = served[0]
        assert (host, port) == ("127.0.0.1", 0)
        assert {"/health", "/search", "/answer", "/mcp"} <= set(paths)
        assert statuses == [200, 400]
        assert "query config" in caplog.text and INDEX.fingerprint in caplog.text

    def test_when_stdio_then_caller_bound_and_server_run(self, wired: FakeTransport) -> None:
        seen: list[tuple[str, Caller]] = []

        def stdio(server: MCPServer) -> None:
            seen.append((server.name, current_caller()))

        bootstrap.run_server(transport="stdio", entitlements=frozenset({"public"}), stdio=stdio)

        assert seen == [("scene-rag-query-api", Caller(frozenset({"public"})))]

    def test_when_uvicorn_serves_then_logging_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))

        bootstrap.serve_with_uvicorn(FastAPI(), "0.0.0.0", 8081)

        assert calls == [{"host": "0.0.0.0", "port": 8081, "log_config": None}]

    def test_when_stdio_runner_called_then_sdk_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        server = MCPServer(name="x")
        monkeypatch.setattr(server, "run", lambda transport: calls.append(transport))

        bootstrap.run_stdio(server)

        assert calls == ["stdio"]

    def test_when_generator_present_then_not_pulled(self) -> None:
        fake = transport()

        bootstrap.pull_models(environ={}, transport=fake)

        assert not any(c[1].endswith("/api/pull") for c in fake.calls)

    def test_when_generator_missing_then_pulled_then_resolved(self) -> None:
        fake = transport(**{"GET /api/tags": {"models": []}})

        def pulled(payload: dict[str, Any] | None) -> dict[str, Any]:
            fake.answers["GET /api/tags"] = TAGS
            return {"status": "success"}

        fake.answers["POST /api/pull"] = pulled

        bootstrap.pull_models(environ={}, transport=fake)

        assert [c[1] for c in fake.calls if c[1].endswith("/api/pull")] == [
            "http://localhost:11434/api/pull"
        ]

    def test_when_pull_fails_then_runtime_error(self) -> None:
        fake = transport(**{"GET /api/tags": {"models": []}, "POST /api/pull": {"status": "error"}})

        with pytest.raises(RuntimeError, match="status 'error'"):
            bootstrap.pull_models(environ={}, transport=fake)


def factory(client: type[object]) -> bootstrap.ClientFactory:
    """A stub class in `MongoClient`'s place: the bootstrap only pings and subscripts it."""
    return cast("bootstrap.ClientFactory", client)


class _Collection:
    def __init__(self, document: dict[str, Any]) -> None:
        self._document = document

    def find_one(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - pymongo's signature
        return self._document
