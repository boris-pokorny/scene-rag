import contextlib
import logging
from collections.abc import Callable
from datetime import UTC
from typing import Any, cast

import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from scene_rag.retrieval.embedding import bootstrap, models, settings
from scene_rag.retrieval.embedding.chunk_reader import Document
from scene_rag.retrieval.embedding.config import EmbeddingConfig, ModelConfig
from scene_rag.retrieval.embedding.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.retrieval.embedding.ollama_embedding import OllamaClient, OllamaEmbeddingModel
from scene_rag.retrieval.embedding.protocols import Dispatcher, EmbeddingModel, SearchIndex
from scene_rag.retrieval.embedding.qdrant_index import QdrantSearchIndex
from scene_rag.retrieval.embedding.report import EmbeddingReport
from scene_rag.retrieval.embedding.settings import EmbeddingSettings, OllamaSettings
from tests.support.embedding_fakes import FakeEmbeddingModel, FakeSearchIndex, FakeTransport

OLLAMA = "http://ollama:11434"
QDRANT = "http://qdrant:6333"
TAGS: dict[str, Any] = {
    "models": [
        {"name": "embeddinggemma:latest", "model": "embeddinggemma:latest", "digest": "ab" * 32}
    ]
}


class CountingScheduler:
    """Stands in for Scheduler: counts passes, fails the ones listed in `failing`."""

    fingerprint = "0123456789ab"

    def __init__(self, failing: frozenset[int] = frozenset()) -> None:
        self.passes = 0
        self.dispatcher: Dispatcher | None = None
        self.model: EmbeddingModel | None = None
        self.config: EmbeddingConfig | None = None
        self._failing = failing

    def run_once(self) -> EmbeddingReport:
        self.passes += 1
        if self.passes in self._failing:
            raise OSError(f"pass {self.passes} failed")
        return EmbeddingReport()


class FakeMongoClient:
    """Stands in for MongoClient: keeps its options, answers or refuses a ping, closes on exit."""

    def __init__(self, uri: str, *, refuse: bool = False, **options: object) -> None:
        self.uri = uri
        self.options = options
        self.entered = False
        self.closed = False
        self._refuse = refuse
        self.admin = self

    def command(self, name: str) -> dict[str, int]:
        if self._refuse:
            raise ServerSelectionTimeoutError("127.0.0.1:1: connection refused")
        return {"ok": 1}

    def __enter__(self) -> FakeMongoClient:
        self.entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True


class FakeCollection:
    """Enough of a pymongo collection for the store constructors and an empty ledger."""

    def __getitem__(self, name: str) -> FakeCollection:
        return self

    def create_index(self, keys: object) -> str:
        return "index"

    def insert_one(self, document: object) -> None:
        return None

    def find_one(self, query: object) -> None:
        return None

    def find(self, *args: object, **kwargs: object) -> list[object]:
        return []


class FakeStoreClient:
    """Enough of a MongoClient for the store constructors: index-creating collections."""

    def __getitem__(self, name: str) -> FakeCollection:
        return FakeCollection()


def ollama_transport(
    *, reachable: bool = True, tags: dict[str, Any] | None = None
) -> FakeTransport:
    transport = FakeTransport(
        {
            "GET /api/version": {"version": "0.40.12"},
            "GET /api/tags": TAGS if tags is None else tags,
            "POST /api/pull": {"status": "success"},
        },
        base=OLLAMA,
    )
    if not reachable:
        transport.unreachable("GET /api/version")
    return transport


def qdrant_transport(*, reachable: bool = True) -> FakeTransport:
    transport = FakeTransport({"GET /": {"version": "1.15.4"}}, base=QDRANT)
    if not reachable:
        transport.unreachable("GET /")
    return transport


type Wire = Callable[..., tuple[CountingScheduler, list[FakeMongoClient]]]


@pytest.fixture
def environment() -> dict[str, str]:
    """Every setting variable."""
    return {
        settings.CHUNK_STORE_URI_ENV: "mongodb://chunks:27017",
        settings.CHUNK_STORE_DATABASE_ENV: "scene_rag_chunks",
        settings.SEARCH_INDEX_URL_ENV: QDRANT,
        settings.SEARCH_INDEX_TIMEOUT_SECONDS_ENV: "12",
        settings.OLLAMA_URL_ENV: OLLAMA,
        settings.OLLAMA_TIMEOUT_SECONDS_ENV: "90",
        settings.BATCH_SIZE_ENV: "8",
        settings.POLL_INTERVAL_SECONDS_ENV: "2.5",
        settings.WORKERS_ENV: "3",
    }


class TestGivenEnvironment:
    """Every setting variable present."""

    def test_when_loaded_then_every_variable_bound(self, environment: dict[str, str]) -> None:
        loaded = bootstrap.load_settings(environment)

        assert loaded == EmbeddingSettings(
            chunk_store_uri="mongodb://chunks:27017",
            chunk_store_database="scene_rag_chunks",
            search_index_url=QDRANT,
            ollama=OllamaSettings(OLLAMA, 90.0),
            search_index_timeout_seconds=12.0,
            batch_size=8,
            poll_interval_seconds=2.5,
            workers=3,
        )

    def test_when_optional_variables_absent_then_defaults(
        self, environment: dict[str, str]
    ) -> None:
        for name in (
            settings.SEARCH_INDEX_TIMEOUT_SECONDS_ENV,
            settings.OLLAMA_URL_ENV,
            settings.OLLAMA_TIMEOUT_SECONDS_ENV,
            settings.POLL_INTERVAL_SECONDS_ENV,
        ):
            del environment[name]
        environment[settings.WORKERS_ENV] = ""
        environment[settings.BATCH_SIZE_ENV] = ""

        loaded = bootstrap.load_settings(environment)

        assert loaded.search_index_timeout_seconds == 30.0
        assert loaded.ollama == OllamaSettings("http://localhost:11434", 120.0)
        assert loaded.batch_size == 32
        assert loaded.poll_interval_seconds is None
        assert loaded.workers == 1

    @pytest.mark.parametrize("variable", [settings.WORKERS_ENV, settings.BATCH_SIZE_ENV])
    @pytest.mark.parametrize("value", ["0", "-2", "two", "1.5"])
    def test_when_count_invalid_then_runtime_error(
        self, environment: dict[str, str], variable: str, value: str
    ) -> None:
        environment[variable] = value

        with pytest.raises(RuntimeError, match="invalid settings") as exc_info:
            bootstrap.load_settings(environment)
        assert variable in str(exc_info.value) and value in str(exc_info.value)

    @pytest.mark.parametrize(
        "missing",
        [
            settings.CHUNK_STORE_URI_ENV,
            settings.CHUNK_STORE_DATABASE_ENV,
            settings.SEARCH_INDEX_URL_ENV,
        ],
    )
    def test_when_variable_removed_then_error_names_it(
        self, environment: dict[str, str], missing: str
    ) -> None:
        del environment[missing]

        with pytest.raises(RuntimeError, match=missing):
            bootstrap.load_settings(environment)

    @pytest.mark.parametrize(
        "variable",
        [
            settings.POLL_INTERVAL_SECONDS_ENV,
            settings.SEARCH_INDEX_TIMEOUT_SECONDS_ENV,
            settings.OLLAMA_TIMEOUT_SECONDS_ENV,
        ],
    )
    @pytest.mark.parametrize("value", ["soon", "nan", "0"])
    def test_when_number_invalid_then_runtime_error(
        self, environment: dict[str, str], variable: str, value: str
    ) -> None:
        environment[variable] = value

        with pytest.raises(RuntimeError, match="invalid settings"):
            bootstrap.load_settings(environment)

    def test_when_ollama_settings_loaded_alone_then_defaults_apply(self) -> None:
        assert bootstrap.load_ollama_settings({}) == OllamaSettings("http://localhost:11434", 120.0)


class TestGivenSystemClock:
    """The production clock."""

    def test_when_now_then_utc_millisecond_precision(self) -> None:
        now = bootstrap.SystemClock().now()
        assert now.tzinfo is UTC
        assert now.microsecond % 1000 == 0


class TestGivenMakeDispatcher:
    """The policy chosen for a worker count, and who closes it."""

    def test_when_one_worker_then_serial_and_nothing_to_close(self) -> None:
        with contextlib.ExitStack() as stack:
            dispatcher = bootstrap.make_dispatcher(1, stack, name="test")

        assert isinstance(dispatcher, SerialDispatcher)

    def test_when_several_workers_then_pool_drained_when_stack_closes(self) -> None:
        with contextlib.ExitStack() as stack:
            dispatcher = bootstrap.make_dispatcher(3, stack, name="test")
            assert isinstance(dispatcher, ThreadDispatcher) and dispatcher.workers == 3
            assert list(dispatcher.map(abs, [-1])) == [1]

        with pytest.raises(RuntimeError, match="shutdown"):
            list(dispatcher.map(abs, [-1]))


class TestGivenBackends:
    """`connect_ollama`, `connect_search_index` and `create_embedding_model` over fakes."""

    @pytest.fixture
    def loaded(self, environment: dict[str, str]) -> EmbeddingSettings:
        return bootstrap.load_settings(environment)

    def test_when_ollama_answers_then_client_bound(self, loaded: EmbeddingSettings) -> None:
        transport = ollama_transport()

        client = bootstrap.connect_ollama(loaded.ollama, transport)

        assert isinstance(client, OllamaClient)
        assert transport.calls[0][:2] == ("GET", OLLAMA + "/api/version")

    def test_when_ollama_unreachable_then_runtime_error_names_variable(
        self, loaded: EmbeddingSettings
    ) -> None:
        with pytest.raises(RuntimeError, match=settings.OLLAMA_URL_ENV) as exc_info:
            bootstrap.connect_ollama(loaded.ollama, ollama_transport(reachable=False))

        assert OLLAMA in str(exc_info.value)

    def test_when_search_index_answers_then_backend_bound(self, loaded: EmbeddingSettings) -> None:
        transport = qdrant_transport()

        index = bootstrap.connect_search_index(loaded, transport)

        assert isinstance(index, QdrantSearchIndex)
        assert transport.calls == [("GET", QDRANT + "/", None, 5.0)]

    def test_when_search_index_unreachable_then_runtime_error_names_variable(
        self, loaded: EmbeddingSettings
    ) -> None:
        with pytest.raises(RuntimeError, match=settings.SEARCH_INDEX_URL_ENV) as exc_info:
            bootstrap.connect_search_index(loaded, qdrant_transport(reachable=False))

        assert QDRANT in str(exc_info.value)

    def test_when_model_created_then_ollama_model_with_resolved_digest(
        self, loaded: EmbeddingSettings
    ) -> None:
        client = bootstrap.connect_ollama(loaded.ollama, ollama_transport())

        model = bootstrap.create_embedding_model(client)

        assert isinstance(model, OllamaEmbeddingModel)
        assert model.model == models.ModelRef("embeddinggemma", "sha256:" + "ab" * 32)
        assert (
            model.fingerprint
            == OllamaEmbeddingModel(client, model.model, EmbeddingConfig()).fingerprint
        )

    def test_when_model_missing_then_runtime_error_with_pull_hint(
        self, loaded: EmbeddingSettings
    ) -> None:
        client = bootstrap.connect_ollama(loaded.ollama, ollama_transport(tags={"models": []}))

        with pytest.raises(RuntimeError, match="pull-models"):
            bootstrap.create_embedding_model(client)


class TestGivenPullModels:
    """The pull-models role over a fake transport."""

    def test_when_model_present_then_nothing_pulled(self, environment: dict[str, str]) -> None:
        transport = ollama_transport()

        bootstrap.pull_models(environ=environment, transport=transport)

        assert [m for m, _, _, _ in transport.calls] == ["GET", "GET"]

    def test_when_model_missing_then_pulled_then_resolved(
        self, environment: dict[str, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        transport = ollama_transport(tags={"models": []})
        calls = 0

        def tags(payload: object) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {"models": []} if calls == 1 else TAGS

        transport.answers["GET /api/tags"] = tags
        caplog.set_level(logging.INFO, logger=bootstrap.__name__)

        bootstrap.pull_models(environ=environment, transport=transport)

        assert ("POST", OLLAMA + "/api/pull") in [(m, u) for m, u, _, _ in transport.calls]
        assert "model embeddinggemma is available as sha256:" in caplog.text

    def test_when_configured_model_differs_then_that_one_pulled(
        self, environment: dict[str, str]
    ) -> None:
        transport = ollama_transport(tags={"models": []})
        transport.answers["POST /api/pull"] = {"status": "error"}
        config = EmbeddingConfig(model=ModelConfig(name="bge-m3"))

        with pytest.raises(RuntimeError, match="bge-m3"):
            bootstrap.pull_models(environ=environment, config=config, transport=transport)

    def test_when_pull_fails_then_runtime_error(self, environment: dict[str, str]) -> None:
        transport = ollama_transport(tags={"models": []})
        transport.unreachable("POST /api/pull")

        with pytest.raises(RuntimeError, match="could not reach"):
            bootstrap.pull_models(environ=environment, transport=transport)

    def test_when_ollama_unreachable_then_runtime_error(self, environment: dict[str, str]) -> None:
        with pytest.raises(RuntimeError, match="not reachable"):
            bootstrap.pull_models(environ=environment, transport=ollama_transport(reachable=False))


class TestGivenWorkerLoop:
    """`run_worker` over a counting scheduler and fake backends, settings from `environment`."""

    @pytest.fixture
    def wire(self, environment: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Wire:
        def start(
            failing: frozenset[int] = frozenset(), poll_interval: str | None = "2.5"
        ) -> tuple[CountingScheduler, list[FakeMongoClient]]:
            if poll_interval is None:
                del environment[settings.POLL_INTERVAL_SECONDS_ENV]
            else:
                environment[settings.POLL_INTERVAL_SECONDS_ENV] = poll_interval
            loaded = bootstrap.load_settings(environment)
            scheduler = CountingScheduler(failing)
            clients: list[FakeMongoClient] = []
            model = FakeEmbeddingModel()

            def connect(uri: str, *, redacted: str) -> FakeMongoClient:
                clients.append(FakeMongoClient(uri))
                return clients[-1]

            def create(
                settings: object,
                chunk_client: object,
                search_index: object,
                model: EmbeddingModel,
                *,
                dispatcher: Dispatcher | None = None,
                config: EmbeddingConfig | None = None,
            ) -> CountingScheduler:
                scheduler.dispatcher = dispatcher
                scheduler.model = model
                scheduler.config = config
                return scheduler

            monkeypatch.setattr(bootstrap, "load_settings", lambda: loaded)
            monkeypatch.setattr(bootstrap, "connect_mongo", connect)
            monkeypatch.setattr(bootstrap, "connect_ollama", lambda settings: object())
            monkeypatch.setattr(
                bootstrap, "connect_search_index", lambda settings: FakeSearchIndex()
            )
            monkeypatch.setattr(
                bootstrap, "create_embedding_model", lambda ollama, config=None: model
            )
            monkeypatch.setattr(bootstrap, "create_scheduler", create)
            return scheduler, clients

        return start

    @staticmethod
    def interrupt_after(count: int) -> tuple[list[float], Callable[[float], None]]:
        """Return a sleep that records its pauses and interrupts on the ``count``-th call."""
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            if len(slept) == count:
                raise KeyboardInterrupt

        return slept, sleep

    def test_when_once_then_one_pass_and_client_closed(self, wire: Wire) -> None:
        scheduler, clients = wire()

        bootstrap.run_worker(once=True)

        assert scheduler.passes == 1
        assert [c.uri for c in clients] == ["mongodb://chunks:27017"]
        assert all(c.entered and c.closed for c in clients)
        assert isinstance(scheduler.model, FakeEmbeddingModel)
        assert scheduler.config == EmbeddingConfig()  # one config, passed through explicitly

    def test_when_workers_configured_then_pool_sized_and_drained_on_exit(
        self, wire: Wire, caplog: pytest.LogCaptureFixture
    ) -> None:
        scheduler, _ = wire()
        caplog.set_level(logging.INFO, logger=bootstrap.__name__)

        bootstrap.run_worker(once=True)

        pool = scheduler.dispatcher
        assert isinstance(pool, ThreadDispatcher) and pool.workers == 3
        with pytest.raises(RuntimeError, match="shutdown"):
            list(pool.map(abs, [-1]))
        assert "embedding fingerprint 0123456789ab (fake fake-embed sha256:" in caplog.text
        assert "3 chunk set(s) at once, 8 text(s) per batch" in caplog.text

    def test_when_single_worker_then_serial_dispatcher(
        self, wire: Wire, environment: dict[str, str]
    ) -> None:
        environment[settings.WORKERS_ENV] = "1"
        scheduler, _ = wire()

        bootstrap.run_worker(once=True)

        assert isinstance(scheduler.dispatcher, SerialDispatcher)

    def test_when_pass_fails_once_then_error_propagates_and_client_closed(self, wire: Wire) -> None:
        _, clients = wire(failing=frozenset({1}))

        with pytest.raises(OSError, match="pass 1 failed"):
            bootstrap.run_worker(once=True)

        assert all(c.closed for c in clients)

    def test_when_passes_keep_failing_then_pause_doubles_until_one_succeeds(
        self, wire: Wire, caplog: pytest.LogCaptureFixture
    ) -> None:
        scheduler, _ = wire(failing=frozenset({1, 2, 3}))
        slept, sleep = self.interrupt_after(5)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert scheduler.passes == 5
        assert slept == [2.5, 5.0, 10.0, 2.5, 2.5]
        assert "retrying in 10.0 s" in caplog.text

    def test_when_backoff_reaches_cap_then_stays_there(self, wire: Wire) -> None:
        wire(failing=frozenset({1, 2, 3}), poll_interval="200")
        slept, sleep = self.interrupt_after(3)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert slept == [200.0, 300.0, 300.0]

    def test_when_failures_outnumber_doublings_then_still_capped(self) -> None:
        """``2.0 ** 1024`` overflows; days of retries must not kill the worker."""
        assert bootstrap._backoff_seconds(30.0, 5000) == 300.0
        assert bootstrap._backoff_seconds(30.0, 1) == 60.0

    def test_when_backend_unavailable_then_error_after_store_checked_and_closed(
        self, wire: Wire, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scheduler, clients = wire()

        def fail(settings: object) -> object:
            raise RuntimeError("Ollama is not reachable")

        monkeypatch.setattr(bootstrap, "connect_ollama", fail)

        with pytest.raises(RuntimeError, match="Ollama is not reachable"):
            bootstrap.run_worker(once=True)

        assert scheduler.passes == 0
        assert len(clients) == 1 and all(c.closed for c in clients)

    def test_when_looping_without_poll_interval_then_error_before_connecting(
        self, wire: Wire
    ) -> None:
        scheduler, clients = wire(poll_interval=None)

        with pytest.raises(RuntimeError, match=settings.POLL_INTERVAL_SECONDS_ENV):
            bootstrap.run_worker(once=False)

        assert scheduler.passes == 0 and clients == []


class TestGivenUnreachableServer:
    """A connection string naming a server that refuses the connectivity check."""

    def test_when_connecting_then_runtime_error_without_password(self) -> None:
        created: list[FakeMongoClient] = []

        def refusing(uri: str, **options: object) -> FakeMongoClient:
            created.append(FakeMongoClient(uri, refuse=True, **options))
            return created[-1]

        with pytest.raises(RuntimeError, match="Could not reach MongoDB") as exc_info:
            bootstrap.connect_mongo(
                "mongodb://app:s3cret@127.0.0.1:1",
                redacted="mongodb://***@127.0.0.1:1",
                client_factory=cast("bootstrap.ClientFactory", refusing),
            )

        assert "s3cret" not in str(exc_info.value)
        assert "mongodb://***@127.0.0.1:1" in str(exc_info.value)
        (client,) = created
        assert client.options == {"tz_aware": True, "serverSelectionTimeoutMS": 2000}

    def test_when_uri_malformed_then_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Could not reach MongoDB"):
            bootstrap.connect_mongo(
                "mongodb://127.0.0.1:notaport", redacted="mongodb://127.0.0.1:notaport"
            )


class TestGivenRealWiring:
    """`create_scheduler` with a fake client, index and model: the reference adapters compose."""

    def test_when_created_then_index_prepared_and_fingerprint_exposed(
        self, environment: dict[str, str]
    ) -> None:
        loaded = bootstrap.load_settings(environment)
        client = cast("MongoClient[Document]", FakeStoreClient())
        search_index = FakeSearchIndex()

        scheduler = bootstrap.create_scheduler(
            loaded, client, search_index, FakeEmbeddingModel("fedcba987654")
        )

        assert scheduler.fingerprint == "fedcba987654"
        assert search_index.describe("chunks_fedcba987654") == models.CollectionInfo(8, "cosine", 0)

    def test_when_index_mismatches_then_runtime_error(self, environment: dict[str, str]) -> None:
        loaded = bootstrap.load_settings(environment)
        client = cast("MongoClient[Document]", FakeStoreClient())
        search_index = FakeSearchIndex()
        search_index.ensure_collection("chunks_fedcba987654", 4, "cosine")

        with pytest.raises(RuntimeError, match="Could not prepare the index"):
            bootstrap.create_scheduler(
                loaded,
                client,
                cast("SearchIndex", search_index),
                FakeEmbeddingModel("fedcba987654"),
            )
