import shutil
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import cast

import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from scene_rag.extraction.segmentation import bootstrap, settings
from scene_rag.extraction.segmentation.asset_registry import Document
from scene_rag.extraction.segmentation.models import SegmentationReport


class CountingService:
    """Stands in for SegmentationService: counts passes, fails the ones listed in `failing`."""

    fingerprint = "0123456789ab"

    def __init__(self, failing: frozenset[int] = frozenset()) -> None:
        self.passes = 0
        self._failing = failing

    def run_once(self) -> SegmentationReport:
        self.passes += 1
        if self.passes in self._failing:
            raise OSError(f"pass {self.passes} failed")
        return SegmentationReport()


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


type Wire = Callable[..., tuple[CountingService, list[FakeMongoClient]]]


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
    """Every setting variable, with ``tmp_path`` as the derivative root."""
    return {
        settings.REGISTRY_URI_ENV: "mongodb://unused:27017",
        settings.REGISTRY_DATABASE_ENV: "scene_rag_test",
        settings.SCENE_STORE_URI_ENV: "mongodb://scenes:27017",
        settings.SCENE_STORE_DATABASE_ENV: "scene_rag_scenes",
        settings.DERIVATIVE_ROOT_ENV: str(tmp_path / "derivatives"),
        settings.FFMPEG_BINARY_ENV: "/opt/ffmpeg/bin/ffmpeg",
        settings.POLL_INTERVAL_SECONDS_ENV: "2.5",
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
        assert loaded.derivative_root == (tmp_path / "derivatives").resolve()
        assert loaded.ffmpeg_binary == "/opt/ffmpeg/bin/ffmpeg"
        assert loaded.poll_interval_seconds == 2.5

    def test_when_optional_variables_absent_then_defaults(
        self, environment: dict[str, str]
    ) -> None:
        del environment[settings.POLL_INTERVAL_SECONDS_ENV]
        del environment[settings.FFMPEG_BINARY_ENV]

        loaded = bootstrap.load_settings(environment)

        assert loaded.poll_interval_seconds is None
        assert loaded.ffmpeg_binary == "ffmpeg"

    @pytest.mark.parametrize(
        "missing",
        [
            settings.REGISTRY_URI_ENV,
            settings.REGISTRY_DATABASE_ENV,
            settings.SCENE_STORE_URI_ENV,
            settings.SCENE_STORE_DATABASE_ENV,
            settings.DERIVATIVE_ROOT_ENV,
        ],
    )
    def test_when_variable_removed_then_error_names_it(
        self, environment: dict[str, str], missing: str
    ) -> None:
        del environment[missing]

        with pytest.raises(RuntimeError, match=missing):
            bootstrap.load_settings(environment)

    @pytest.mark.parametrize("value", ["soon", "nan", "0"])
    def test_when_poll_interval_invalid_then_runtime_error(
        self, environment: dict[str, str], value: str
    ) -> None:
        environment[settings.POLL_INTERVAL_SECONDS_ENV] = value

        with pytest.raises(RuntimeError, match="invalid settings"):
            bootstrap.load_settings(environment)


class TestGivenSystemClock:
    """The production clock."""

    def test_when_now_then_utc_millisecond_precision(self) -> None:
        now = bootstrap.SystemClock().now()
        assert now.tzinfo is UTC
        assert now.microsecond % 1000 == 0


class TestGivenWorkerLoop:
    """`run_worker` over a counting service and fake clients, settings from `environment`."""

    @pytest.fixture
    def wire(self, environment: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Wire:
        def start(
            failing: frozenset[int] = frozenset(), poll_interval: str | None = "2.5"
        ) -> tuple[CountingService, list[FakeMongoClient]]:
            if poll_interval is None:
                del environment[settings.POLL_INTERVAL_SECONDS_ENV]
            else:
                environment[settings.POLL_INTERVAL_SECONDS_ENV] = poll_interval
            loaded = bootstrap.load_settings(environment)
            service = CountingService(failing)
            clients: list[FakeMongoClient] = []

            def connect(uri: str, *, redacted: str) -> FakeMongoClient:
                clients.append(FakeMongoClient(uri))
                return clients[-1]

            monkeypatch.setattr(bootstrap, "load_settings", lambda: loaded)
            monkeypatch.setattr(shutil, "which", lambda binary: binary)
            monkeypatch.setattr(bootstrap, "connect_mongo", connect)
            monkeypatch.setattr(
                bootstrap, "create_service", lambda settings, registry, scenes: service
            )
            return service, clients

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

    def test_when_once_then_one_pass_and_both_clients_closed(self, wire: Wire) -> None:
        service, clients = wire()

        bootstrap.run_worker(once=True)

        assert service.passes == 1
        assert [c.uri for c in clients] == ["mongodb://unused:27017", "mongodb://scenes:27017"]
        assert all(c.entered and c.closed for c in clients)

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
        service, _ = wire(failing=frozenset({1, 2, 3}))
        slept, sleep = self.interrupt_after(5)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert service.passes == 5
        assert slept == [2.5, 5.0, 10.0, 2.5, 2.5]
        assert "retrying in 10.0 s" in caplog.text

    def test_when_backoff_reaches_cap_then_stays_there(self, wire: Wire) -> None:
        wire(failing=frozenset({1, 2, 3}), poll_interval="200")
        slept, sleep = self.interrupt_after(3)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert slept == [200.0, 300.0, 300.0]

    def test_when_ffmpeg_missing_then_error_before_connecting(
        self, wire: Wire, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service, clients = wire()
        monkeypatch.setattr(shutil, "which", lambda binary: None)

        with pytest.raises(RuntimeError, match=settings.FFMPEG_BINARY_ENV) as exc_info:
            bootstrap.run_worker(once=True)

        assert "/opt/ffmpeg/bin/ffmpeg" in str(exc_info.value)
        assert service.passes == 0 and clients == []

    def test_when_looping_without_poll_interval_then_error_before_connecting(
        self, wire: Wire
    ) -> None:
        service, clients = wire(poll_interval=None)

        with pytest.raises(RuntimeError, match=settings.POLL_INTERVAL_SECONDS_ENV):
            bootstrap.run_worker(once=False)

        assert service.passes == 0 and clients == []


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
    """`create_service` with fake clients: the reference backends compose."""

    def test_when_created_then_fingerprint_computed_and_root_and_workspace_created(
        self, environment: dict[str, str], tmp_path: Path
    ) -> None:
        loaded = bootstrap.load_settings(environment)
        client = cast("MongoClient[Document]", FakeStoreClient())

        service = bootstrap.create_service(loaded, client, client)

        assert len(service.fingerprint) == 12
        assert (tmp_path / "derivatives" / bootstrap.WORKSPACE_DIRECTORY).is_dir()


class FakeStoreClient:
    """Enough of a MongoClient for the store constructors: index-creating collections."""

    def __getitem__(self, database: str) -> FakeStoreClient:
        return self

    def create_index(self, keys: object) -> str:
        return "index"
