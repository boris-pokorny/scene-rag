"""Composition of the scene-api role: settings, connections, the served application."""

import logging
from collections.abc import Mapping
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from scene_rag.extraction.scene_api import bootstrap, settings
from scene_rag.extraction.scene_api.asset_registry import Document
from scene_rag.extraction.scene_api.service import SceneCatalogService
from tests.support.scene_api_fakes import make_scene_api_harness


class FakeMongoClient:
    """Stands in for MongoClient: keeps its options, answers or refuses a ping, closes on exit."""

    def __init__(
        self, uri: str, options: Mapping[str, object] | None = None, *, refuse: bool = False
    ) -> None:
        self.uri = uri
        self.options = dict(options or {})
        self.closed = False
        self._refuse = refuse
        self.admin = self

    def command(self, name: str) -> dict[str, int]:
        if self._refuse:
            raise ServerSelectionTimeoutError("127.0.0.1:1: connection refused")
        return {"ok": 1}

    def __enter__(self) -> FakeMongoClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True


class FakeCollectionClient:
    """Enough of a MongoClient for the reader constructors: databases of collections."""

    def __getitem__(self, name: str) -> FakeCollectionClient:
        return self


@pytest.fixture
def environment() -> dict[str, str]:
    return {
        settings.REGISTRY_URI_ENV: "mongodb://registry:27017",
        settings.REGISTRY_DATABASE_ENV: "scene_rag_registry",
        settings.SCENE_STORE_URI_ENV: "mongodb://scenes:27017",
        settings.SCENE_STORE_DATABASE_ENV: "scene_rag_scenes",
        settings.ENRICHMENT_STORE_URI_ENV: "mongodb://enrichments:27017",
        settings.ENRICHMENT_STORE_DATABASE_ENV: "scene_rag_enrichments",
        settings.HOST_ENV: "0.0.0.0",
        settings.PORT_ENV: "9000",
        settings.PAGE_SIZE_ENV: "25",
    }


class TestGivenEnvironment:
    """Binding settings from variables."""

    def test_when_loaded_then_every_variable_bound(self, environment: dict[str, str]) -> None:
        loaded = bootstrap.load_settings(environment)

        assert loaded.registry_uri == "mongodb://registry:27017"
        assert loaded.registry_database == "scene_rag_registry"
        assert loaded.scene_store_uri == "mongodb://scenes:27017"
        assert loaded.scene_store_database == "scene_rag_scenes"
        assert loaded.enrichment_store_uri == "mongodb://enrichments:27017"
        assert loaded.enrichment_store_database == "scene_rag_enrichments"
        assert (loaded.host, loaded.port, loaded.page_size) == ("0.0.0.0", 9000, 25)

    def test_when_optional_variables_absent_then_defaults(
        self, environment: dict[str, str]
    ) -> None:
        del environment[settings.HOST_ENV]
        environment[settings.PORT_ENV] = ""
        del environment[settings.PAGE_SIZE_ENV]

        loaded = bootstrap.load_settings(environment)

        assert (loaded.host, loaded.port, loaded.page_size) == ("127.0.0.1", 8080, 100)

    @pytest.mark.parametrize(
        "missing",
        [
            settings.REGISTRY_URI_ENV,
            settings.REGISTRY_DATABASE_ENV,
            settings.SCENE_STORE_URI_ENV,
            settings.SCENE_STORE_DATABASE_ENV,
            settings.ENRICHMENT_STORE_URI_ENV,
            settings.ENRICHMENT_STORE_DATABASE_ENV,
        ],
    )
    def test_when_variable_removed_then_error_names_it(
        self, environment: dict[str, str], missing: str
    ) -> None:
        del environment[missing]

        with pytest.raises(RuntimeError, match=missing):
            bootstrap.load_settings(environment)

    @pytest.mark.parametrize(
        "variable, value",
        [
            (settings.PORT_ENV, "-1"),
            (settings.PORT_ENV, "70000"),
            (settings.PORT_ENV, "eighty"),
            (settings.PAGE_SIZE_ENV, "0"),
            (settings.PAGE_SIZE_ENV, "1.5"),
        ],
    )
    def test_when_number_invalid_then_runtime_error(
        self, environment: dict[str, str], variable: str, value: str
    ) -> None:
        environment[variable] = value

        with pytest.raises(RuntimeError, match="invalid settings"):
            bootstrap.load_settings(environment)

    def test_when_process_environment_used_then_read(
        self, environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in environment.items():
            monkeypatch.setenv(name, value)

        assert bootstrap.load_settings().port == 9000


class TestGivenMongo:
    """Connecting to the stores."""

    def test_when_server_answers_then_client_returned_with_options(self) -> None:
        made: list[FakeMongoClient] = []

        def factory(uri: str, **options: object) -> MongoClient[Document]:
            made.append(FakeMongoClient(uri, options))
            return cast(MongoClient[Document], made[-1])

        client = bootstrap.connect_mongo(
            "mongodb://u:p@h:27017", redacted="mongodb://***@h:27017", client_factory=factory
        )

        assert cast(object, client) is made[0]
        assert made[0].options == {"tz_aware": True, "serverSelectionTimeoutMS": 2000}

    def test_when_server_refuses_then_runtime_error_names_redacted_uri(self) -> None:
        def factory(uri: str, **options: object) -> MongoClient[Document]:
            return cast(MongoClient[Document], FakeMongoClient(uri, options, refuse=True))

        with pytest.raises(RuntimeError, match="mongodb://\\*\\*\\*@h:27017") as exc_info:
            bootstrap.connect_mongo(
                "mongodb://u:p@h:27017", redacted="mongodb://***@h:27017", client_factory=factory
            )
        assert "u:p" not in str(exc_info.value)

    def test_when_uri_malformed_then_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Could not reach"):
            bootstrap.connect_mongo("mongodb://h:notaport", redacted="mongodb://h:notaport")


class TestGivenWiring:
    """The service and the server."""

    def test_when_catalog_created_then_mongo_readers_over_named_databases(self) -> None:
        loaded = bootstrap.load_settings(
            {
                settings.REGISTRY_URI_ENV: "mongodb://x",
                settings.REGISTRY_DATABASE_ENV: "r",
                settings.SCENE_STORE_URI_ENV: "mongodb://x",
                settings.SCENE_STORE_DATABASE_ENV: "s",
                settings.ENRICHMENT_STORE_URI_ENV: "mongodb://x",
                settings.ENRICHMENT_STORE_DATABASE_ENV: "e",
                settings.PAGE_SIZE_ENV: "7",
            }
        )
        client = cast(MongoClient[Document], FakeCollectionClient())

        catalog = bootstrap.create_catalog(loaded, client, client, client)

        assert isinstance(catalog, SceneCatalogService)
        assert catalog._page_size == 7
        assert catalog._page_fill == bootstrap.PAGE_FILL

    def test_when_service_created_then_page_size_applied(self) -> None:
        harness = make_scene_api_harness()

        service = bootstrap.create_service(
            harness.assets,
            harness.scene_sets,
            harness.enrichments,
            harness.derivatives,
            page_size=3,
            page_fill=2,
        )

        assert (service._page_size, service._page_fill) == (3, 2)

    def test_when_server_run_then_app_served_and_clients_closed(
        self,
        environment: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        clients: list[FakeMongoClient] = []

        def connect(uri: str, *, redacted: str) -> MongoClient[Document]:
            clients.append(FakeMongoClient(uri))
            return cast(MongoClient[Document], clients[-1])

        monkeypatch.setattr(bootstrap, "connect_mongo", connect)
        monkeypatch.setattr(
            bootstrap, "create_catalog", lambda *_: make_scene_api_harness().service
        )
        served: list[tuple[FastAPI, str, int]] = []

        def serve(app: FastAPI, host: str, port: int) -> None:
            served.append((app, host, port))
            assert not any(client.closed for client in clients)
            assert TestClient(app).get("/health").json() == {"status": "ok"}

        with caplog.at_level(logging.INFO):
            bootstrap.run_server(serve=serve)

        assert [c.uri for c in clients] == [
            "mongodb://registry:27017",
            "mongodb://scenes:27017",
            "mongodb://enrichments:27017",
        ]
        assert [(host, port) for _, host, port in served] == [("0.0.0.0", 9000)]
        assert all(client.closed for client in clients)
        assert "http://0.0.0.0:9000" in caplog.text

    def test_when_uvicorn_serves_then_told_not_to_configure_logging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "scene_rag.extraction.scene_api.bootstrap.uvicorn.run",
            lambda app, **kw: calls.append({"app": app, **kw}),
        )
        app = FastAPI()

        bootstrap.serve_with_uvicorn(app, "127.0.0.1", 0)

        assert calls == [{"app": app, "host": "127.0.0.1", "port": 0, "log_config": None}]
