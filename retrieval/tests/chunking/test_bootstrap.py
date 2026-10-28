import contextlib
import logging
from collections.abc import Callable
from datetime import UTC
from typing import Any, cast

import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from scene_rag.retrieval.chunking import bootstrap, settings
from scene_rag.retrieval.chunking.chunk_store import Document
from scene_rag.retrieval.chunking.config import ChunkingConfig
from scene_rag.retrieval.chunking.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.retrieval.chunking.per_scene import PerSceneChunker
from scene_rag.retrieval.chunking.protocols import Chunker, Dispatcher, SceneApiError
from scene_rag.retrieval.chunking.report import ChunkingReport
from scene_rag.retrieval.chunking.scene_api_client import SceneApiClient
from scene_rag.retrieval.chunking.settings import ChunkingSettings
from tests.support.chunking_fakes import FakeChunker, FakeSceneRecordReader


class CountingScheduler:
    """Stands in for Scheduler: counts passes, fails the ones listed in `failing`."""

    fingerprint = "0123456789ab"

    def __init__(self, failing: frozenset[int] = frozenset()) -> None:
        self.passes = 0
        self.dispatcher: Dispatcher | None = None
        self.chunker: Chunker | None = None
        self._failing = failing

    def run_once(self) -> ChunkingReport:
        self.passes += 1
        if self.passes in self._failing:
            raise OSError(f"pass {self.passes} failed")
        return ChunkingReport()


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


class FakeStoreClient:
    """Enough of a MongoClient for the store constructor: index-creating collections."""

    def __getitem__(self, database: str) -> FakeStoreClient:
        return self

    def create_index(self, keys: object) -> str:
        return "index"


class FakeTransport:
    """JsonTransport answering a healthy scene-api, or failing."""

    def __init__(self, reachable: bool = True, status: str = "ok") -> None:
        self.reachable = reachable
        self.status = status
        self.urls: list[str] = []

    def get(self, url: str, timeout: float) -> dict[str, Any]:
        self.urls.append(url)
        if not self.reachable:
            raise SceneApiError(f"could not reach {url}")
        return {"status": self.status}


type Wire = Callable[..., tuple[CountingScheduler, list[FakeMongoClient]]]


@pytest.fixture
def environment() -> dict[str, str]:
    """Every setting variable."""
    return {
        settings.SCENE_API_URL_ENV: "http://scene-api:8080",
        settings.SCENE_API_TIMEOUT_SECONDS_ENV: "12",
        settings.CHUNK_STORE_URI_ENV: "mongodb://chunks:27017",
        settings.CHUNK_STORE_DATABASE_ENV: "scene_rag_chunks",
        settings.POLL_INTERVAL_SECONDS_ENV: "2.5",
        settings.WORKERS_ENV: "3",
    }


class TestGivenEnvironment:
    """Every setting variable present."""

    def test_when_loaded_then_every_variable_bound(self, environment: dict[str, str]) -> None:
        loaded = bootstrap.load_settings(environment)

        assert loaded == ChunkingSettings(
            scene_api_url="http://scene-api:8080",
            chunk_store_uri="mongodb://chunks:27017",
            chunk_store_database="scene_rag_chunks",
            scene_api_timeout_seconds=12.0,
            poll_interval_seconds=2.5,
            workers=3,
        )

    def test_when_optional_variables_absent_then_defaults(
        self, environment: dict[str, str]
    ) -> None:
        del environment[settings.SCENE_API_TIMEOUT_SECONDS_ENV]
        del environment[settings.POLL_INTERVAL_SECONDS_ENV]
        environment[settings.WORKERS_ENV] = ""

        loaded = bootstrap.load_settings(environment)

        assert loaded.scene_api_timeout_seconds == 30.0
        assert loaded.poll_interval_seconds is None
        assert loaded.workers == 1

    @pytest.mark.parametrize("value", ["0", "-2", "two", "1.5"])
    def test_when_worker_count_invalid_then_runtime_error(
        self, environment: dict[str, str], value: str
    ) -> None:
        environment[settings.WORKERS_ENV] = value

        with pytest.raises(RuntimeError, match="invalid settings") as exc_info:
            bootstrap.load_settings(environment)
        assert settings.WORKERS_ENV in str(exc_info.value) and value in str(exc_info.value)

    @pytest.mark.parametrize(
        "missing",
        [
            settings.SCENE_API_URL_ENV,
            settings.CHUNK_STORE_URI_ENV,
            settings.CHUNK_STORE_DATABASE_ENV,
        ],
    )
    def test_when_variable_removed_then_error_names_it(
        self, environment: dict[str, str], missing: str
    ) -> None:
        del environment[missing]

        with pytest.raises(RuntimeError, match=missing):
            bootstrap.load_settings(environment)

    @pytest.mark.parametrize(
        "variable", [settings.POLL_INTERVAL_SECONDS_ENV, settings.SCENE_API_TIMEOUT_SECONDS_ENV]
    )
    @pytest.mark.parametrize("value", ["soon", "nan", "0"])
    def test_when_number_invalid_then_runtime_error(
        self, environment: dict[str, str], variable: str, value: str
    ) -> None:
        environment[variable] = value

        with pytest.raises(RuntimeError, match="invalid settings"):
            bootstrap.load_settings(environment)


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


class TestGivenSceneApi:
    """`connect_scene_api` over a fake transport."""

    @pytest.fixture
    def loaded(self, environment: dict[str, str]) -> ChunkingSettings:
        return bootstrap.load_settings(environment)

    def test_when_api_answers_then_client_bound_to_url_and_timeout(
        self, loaded: ChunkingSettings
    ) -> None:
        transport = FakeTransport()

        client = bootstrap.connect_scene_api(loaded, transport)

        assert isinstance(client, SceneApiClient)
        assert transport.urls == ["http://scene-api:8080/health"]

    @pytest.mark.parametrize(
        "transport", [FakeTransport(reachable=False), FakeTransport(status="no")]
    )
    def test_when_api_unreachable_then_runtime_error_names_variable(
        self, loaded: ChunkingSettings, transport: FakeTransport
    ) -> None:
        with pytest.raises(RuntimeError, match=settings.SCENE_API_URL_ENV) as exc_info:
            bootstrap.connect_scene_api(loaded, transport)

        assert "http://scene-api:8080" in str(exc_info.value)


class TestGivenChunkerRegistration:
    """`create_chunker` composes the reference strategy."""

    def test_when_created_then_per_scene_with_fingerprint(self) -> None:
        chunker = bootstrap.create_chunker()

        assert isinstance(chunker, PerSceneChunker)
        assert chunker.fingerprint == PerSceneChunker(ChunkingConfig()).fingerprint


class TestGivenWorkerLoop:
    """`run_worker` over a counting scheduler and fake clients, settings from `environment`."""

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
            chunker = FakeChunker()

            def connect(uri: str, *, redacted: str) -> FakeMongoClient:
                clients.append(FakeMongoClient(uri))
                return clients[-1]

            def create(
                settings: object,
                chunk_client: object,
                reader: object,
                *,
                chunker: Chunker,
                dispatcher: Dispatcher | None = None,
            ) -> CountingScheduler:
                scheduler.dispatcher = dispatcher
                scheduler.chunker = chunker
                return scheduler

            monkeypatch.setattr(bootstrap, "load_settings", lambda: loaded)
            monkeypatch.setattr(bootstrap, "connect_mongo", connect)
            monkeypatch.setattr(
                bootstrap, "connect_scene_api", lambda settings: FakeSceneRecordReader()
            )
            monkeypatch.setattr(bootstrap, "create_chunker", lambda: chunker)
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
        assert isinstance(scheduler.chunker, FakeChunker)

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
        assert "chunking fingerprint 0123456789ab (fake); 3 scene set(s) at once" in caplog.text

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

    def test_when_scene_api_unavailable_then_error_after_store_checked_and_closed(
        self, wire: Wire, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scheduler, clients = wire()

        def fail(settings: object) -> FakeSceneRecordReader:
            raise RuntimeError("The scene-api is not reachable")

        monkeypatch.setattr(bootstrap, "connect_scene_api", fail)

        with pytest.raises(RuntimeError, match="scene-api is not reachable"):
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
    """`create_scheduler` with a fake client, reader and chunker: the reference adapters compose."""

    def test_when_created_then_fingerprint_exposed(self, environment: dict[str, str]) -> None:
        loaded = bootstrap.load_settings(environment)
        client = cast("MongoClient[Document]", FakeStoreClient())

        scheduler = bootstrap.create_scheduler(
            loaded, client, FakeSceneRecordReader(), chunker=FakeChunker("fedcba987654")
        )

        assert scheduler.fingerprint == "fedcba987654"
