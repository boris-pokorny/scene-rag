import contextlib
import logging
from collections.abc import Callable, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any, cast

import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from scene_rag.extraction.enrichment import bootstrap, settings, whisper_loader
from scene_rag.extraction.enrichment.asset_registry import Document
from scene_rag.extraction.enrichment.config import EnrichmentConfig
from scene_rag.extraction.enrichment.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.extraction.enrichment.models import EnrichmentReport, ModelRef
from scene_rag.extraction.enrichment.protocols import Dispatcher, Enricher
from scene_rag.extraction.enrichment.settings import EnrichmentSettings
from tests.support.enrichment_fakes import FakeEnricher, FakeTranscriber


class CountingScheduler:
    """Stands in for Scheduler: counts passes, fails the ones listed in `failing`."""

    fingerprints = {"description": "0123456789ab", "transcript": "fedcba987654"}

    def __init__(self, failing: frozenset[int] = frozenset()) -> None:
        self.passes = 0
        self.dispatcher: Dispatcher | None = None
        self.enrichers: Sequence[Enricher] = ()
        self._failing = failing

    def run_once(self) -> EnrichmentReport:
        self.passes += 1
        if self.passes in self._failing:
            raise OSError(f"pass {self.passes} failed")
        return EnrichmentReport()


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
    """Enough of a MongoClient for the store constructors: index-creating collections."""

    def __getitem__(self, database: str) -> FakeStoreClient:
        return self

    def create_index(self, keys: object) -> str:
        return "index"


class FakeTransport:
    """JsonTransport answering a healthy server holding the vision model, or failing."""

    def __init__(self, reachable: bool = True, models: Sequence[str] = ("qwen2.5vl:3b",)) -> None:
        self.reachable = reachable
        self.models = models

    def get(self, url: str, timeout: float) -> dict[str, Any]:
        if not self.reachable:
            from scene_rag.extraction.enrichment.ollama_transport import OllamaError

            raise OllamaError(f"could not reach {url}")
        if url.endswith("/api/version"):
            return {"version": "0.12.0"}
        return {"models": [{"name": m, "model": m, "digest": "ab" * 32} for m in self.models]}

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"message": {"content": "A talk."}}


type Wire = Callable[..., tuple[CountingScheduler, list[FakeMongoClient]]]


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
    """Every setting variable, with ``tmp_path`` as the model cache."""
    return {
        settings.REGISTRY_URI_ENV: "mongodb://unused:27017",
        settings.REGISTRY_DATABASE_ENV: "scene_rag_test",
        settings.SCENE_STORE_URI_ENV: "mongodb://scenes:27017",
        settings.SCENE_STORE_DATABASE_ENV: "scene_rag_scenes",
        settings.ENRICHMENT_STORE_URI_ENV: "mongodb://enrichments:27017",
        settings.ENRICHMENT_STORE_DATABASE_ENV: "scene_rag_enrichments",
        settings.OLLAMA_URL_ENV: "http://ollama:11434",
        settings.OLLAMA_TIMEOUT_SECONDS_ENV: "120",
        settings.WHISPER_CACHE_DIR_ENV: str(tmp_path / "models"),
        settings.WHISPER_CPU_THREADS_ENV: "4",
        settings.POLL_INTERVAL_SECONDS_ENV: "2.5",
        settings.WORKERS_ENV: "3",
    }


class TestGivenEnvironment:
    """Every setting variable present."""

    def test_when_loaded_then_every_variable_bound(
        self, environment: dict[str, str], tmp_path: Path
    ) -> None:
        loaded = bootstrap.load_settings(environment)

        assert loaded.registry_uri == "mongodb://unused:27017"
        assert loaded.registry_database == "scene_rag_test"
        assert loaded.scene_store_uri == "mongodb://scenes:27017"
        assert loaded.scene_store_database == "scene_rag_scenes"
        assert loaded.enrichment_store_uri == "mongodb://enrichments:27017"
        assert loaded.enrichment_store_database == "scene_rag_enrichments"
        assert loaded.ollama_url == "http://ollama:11434"
        assert loaded.ollama_timeout_seconds == 120.0
        assert loaded.whisper_cache_dir == (tmp_path / "models").resolve()
        assert loaded.whisper_cpu_threads == 4
        assert loaded.poll_interval_seconds == 2.5
        assert loaded.workers == 3

    def test_when_optional_variables_absent_then_defaults(
        self, environment: dict[str, str]
    ) -> None:
        for name in (
            settings.OLLAMA_URL_ENV,
            settings.OLLAMA_TIMEOUT_SECONDS_ENV,
            settings.WHISPER_CACHE_DIR_ENV,
            settings.POLL_INTERVAL_SECONDS_ENV,
        ):
            del environment[name]
        environment[settings.WHISPER_CPU_THREADS_ENV] = ""
        environment[settings.WORKERS_ENV] = ""

        loaded = bootstrap.load_settings(environment)

        assert loaded.ollama_url == "http://localhost:11434"
        assert loaded.ollama_timeout_seconds == 600.0
        assert loaded.whisper_cache_dir is None
        assert loaded.whisper_cpu_threads == 0
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

    @pytest.mark.parametrize("value", ["-1", "many"])
    def test_when_thread_count_invalid_then_runtime_error(
        self, environment: dict[str, str], value: str
    ) -> None:
        environment[settings.WHISPER_CPU_THREADS_ENV] = value

        with pytest.raises(RuntimeError, match=settings.WHISPER_CPU_THREADS_ENV):
            bootstrap.load_settings(environment)

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
        "variable", [settings.POLL_INTERVAL_SECONDS_ENV, settings.OLLAMA_TIMEOUT_SECONDS_ENV]
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


class TestGivenModels:
    """`create_enrichers` over a fake Ollama transport and a fake speech model loader."""

    @pytest.fixture
    def loaded(self, environment: dict[str, str]) -> EnrichmentSettings:
        return bootstrap.load_settings(environment)

    @pytest.fixture
    def speech(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        loads: list[dict[str, object]] = []

        def load_transcriber(
            config: object, *, cache_dir: Path | None, cpu_threads: int
        ) -> FakeTranscriber:
            loads.append({"cache_dir": cache_dir, "cpu_threads": cpu_threads})
            return FakeTranscriber()

        monkeypatch.setattr(whisper_loader, "load_transcriber", load_transcriber)
        return loads

    def test_when_models_available_then_two_enrichers_with_resolved_models(
        self, loaded: EnrichmentSettings, speech: list[dict[str, object]], tmp_path: Path
    ) -> None:
        enrichers = bootstrap.create_enrichers(loaded, transport=FakeTransport())

        assert [e.name for e in enrichers] == ["description", "transcript"]
        assert enrichers[0].model == ModelRef("qwen2.5vl:3b", "sha256:" + "ab" * 32)
        assert all(len(e.fingerprint) == 12 for e in enrichers)
        assert speech == [{"cache_dir": (tmp_path / "models").resolve(), "cpu_threads": 4}]

    def test_when_ollama_unreachable_then_runtime_error_names_variable(
        self, loaded: EnrichmentSettings, speech: list[dict[str, object]]
    ) -> None:
        with pytest.raises(RuntimeError, match=settings.OLLAMA_URL_ENV) as exc_info:
            bootstrap.create_enrichers(loaded, transport=FakeTransport(reachable=False))

        assert "http://ollama:11434" in str(exc_info.value) and speech == []

    def test_when_vision_model_missing_then_runtime_error_with_pull_hint(
        self, loaded: EnrichmentSettings, speech: list[dict[str, object]]
    ) -> None:
        with pytest.raises(RuntimeError, match="ollama pull qwen2.5vl:3b"):
            bootstrap.create_enrichers(loaded, transport=FakeTransport(models=("llama3.2:latest",)))

        assert speech == []


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
            enrichers = (FakeEnricher("description"), FakeEnricher("transcript", "fedcba987654"))

            def connect(uri: str, *, redacted: str) -> FakeMongoClient:
                clients.append(FakeMongoClient(uri))
                return clients[-1]

            def create(
                settings: object,
                registry: object,
                scenes: object,
                enrichments: object,
                *,
                enrichers: Sequence[Enricher],
                dispatcher: Dispatcher | None = None,
            ) -> CountingScheduler:
                scheduler.dispatcher = dispatcher
                scheduler.enrichers = enrichers
                return scheduler

            monkeypatch.setattr(bootstrap, "load_settings", lambda: loaded)
            monkeypatch.setattr(bootstrap, "create_enrichers", lambda settings: enrichers)
            monkeypatch.setattr(bootstrap, "connect_mongo", connect)
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

    def test_when_once_then_one_pass_and_every_client_closed(self, wire: Wire) -> None:
        scheduler, clients = wire()

        bootstrap.run_worker(once=True)

        assert scheduler.passes == 1
        assert [c.uri for c in clients] == [
            "mongodb://unused:27017",
            "mongodb://scenes:27017",
            "mongodb://enrichments:27017",
        ]
        assert all(c.entered and c.closed for c in clients)
        assert [e.name for e in scheduler.enrichers] == ["description", "transcript"]

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
        assert (
            "description=0123456789ab transcript=fedcba987654; 3 scene set(s) at once"
            in caplog.text
        )

    def test_when_single_worker_then_serial_dispatcher(
        self, wire: Wire, environment: dict[str, str]
    ) -> None:
        environment[settings.WORKERS_ENV] = "1"
        scheduler, _ = wire()

        bootstrap.run_worker(once=True)

        assert isinstance(scheduler.dispatcher, SerialDispatcher)

    def test_when_pass_fails_once_then_error_propagates_and_clients_closed(
        self, wire: Wire
    ) -> None:
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

    def test_when_models_unavailable_then_error_after_stores_checked_and_closed(
        self, wire: Wire, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scheduler, clients = wire()

        def fail(settings: object) -> tuple[Enricher, ...]:
            raise RuntimeError("Ollama is not reachable")

        monkeypatch.setattr(bootstrap, "create_enrichers", fail)

        with pytest.raises(RuntimeError, match="Ollama is not reachable"):
            bootstrap.run_worker(once=True)

        assert scheduler.passes == 0
        assert len(clients) == 3 and all(c.closed for c in clients)

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
    """`create_scheduler` with fake clients and enrichers: the reference adapters compose."""

    def test_when_created_then_fingerprints_exposed(self, environment: dict[str, str]) -> None:
        loaded = bootstrap.load_settings(environment)
        client = cast("MongoClient[Document]", FakeStoreClient())
        enrichers = (FakeEnricher("description"), FakeEnricher("transcript", "fedcba987654"))

        scheduler = bootstrap.create_scheduler(
            loaded, client, client, client, enrichers=enrichers, config=EnrichmentConfig()
        )

        assert scheduler.fingerprints == {
            "description": "0123456789ab",
            "transcript": "fedcba987654",
        }
