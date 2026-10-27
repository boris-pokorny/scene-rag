import dataclasses
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import cast

import pytest
from pymongo.errors import ServerSelectionTimeoutError

from scene_rag.extraction.ingestion import bootstrap, settings
from scene_rag.extraction.ingestion.models import IngestionReport
from scene_rag.extraction.ingestion.settings import IngestionSettings


class CountingService:
    """Stands in for IngestionService: counts passes, fails the ones listed in `failing`."""

    fingerprint = "0123456789ab"

    def __init__(self, failing: frozenset[int] = frozenset()) -> None:
        self.passes = 0
        self._failing = failing

    def run_once(self) -> IngestionReport:
        self.passes += 1
        if self.passes in self._failing:
            raise OSError(f"pass {self.passes} failed")
        return IngestionReport()


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


type Wire = Callable[..., tuple[CountingService, FakeMongoClient]]


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
    """Every setting variable, with ``tmp_path`` as the source root."""
    return {
        settings.REGISTRY_URI_ENV: "mongodb://unused:27017",
        settings.REGISTRY_DATABASE_ENV: "scene_rag_test",
        settings.SOURCE_ROOT_ENV: str(tmp_path),
        settings.SOURCE_ID_ENV: "local",
        settings.SOURCE_ENTITLEMENTS_ENV: "public, team-x,",
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
        assert loaded.source_root == tmp_path.resolve()
        assert loaded.source_id == "local"
        assert loaded.source_entitlements == frozenset({"public", "team-x"})
        assert loaded.poll_interval_seconds == 2.5

    def test_when_poll_interval_absent_then_none(self, environment: dict[str, str]) -> None:
        del environment[settings.POLL_INTERVAL_SECONDS_ENV]

        assert bootstrap.load_settings(environment).poll_interval_seconds is None

    @pytest.mark.parametrize(
        "missing",
        [
            settings.REGISTRY_URI_ENV,
            settings.REGISTRY_DATABASE_ENV,
            settings.SOURCE_ROOT_ENV,
            settings.SOURCE_ID_ENV,
            settings.SOURCE_ENTITLEMENTS_ENV,
        ],
    )
    def test_when_variable_removed_then_error_names_it(
        self, environment: dict[str, str], missing: str
    ) -> None:
        del environment[missing]

        with pytest.raises(RuntimeError, match=missing):
            bootstrap.load_settings(environment)

    @pytest.mark.parametrize(
        ("variable", "value", "message"),
        [
            (settings.SOURCE_ROOT_ENV, "missing-directory", "source_root is not a directory"),
            (settings.POLL_INTERVAL_SECONDS_ENV, "soon", "invalid settings"),
            (settings.POLL_INTERVAL_SECONDS_ENV, "nan", "poll_interval_seconds"),
            (settings.POLL_INTERVAL_SECONDS_ENV, "inf", "poll_interval_seconds"),
            (settings.SOURCE_ENTITLEMENTS_ENV, " , ", "source_entitlements"),
        ],
    )
    def test_when_value_invalid_then_runtime_error(
        self, environment: dict[str, str], variable: str, value: str, message: str
    ) -> None:
        environment[variable] = value

        with pytest.raises(RuntimeError, match=message):
            bootstrap.load_settings(environment)


class TestGivenValidSettings:
    """Settings already bound and validated."""

    @pytest.fixture
    def valid(self, environment: dict[str, str]) -> IngestionSettings:
        return bootstrap.load_settings(environment)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda s: dataclasses.replace(s, registry_uri=""), "registry_uri"),
            (lambda s: dataclasses.replace(s, registry_database=""), "registry_database"),
            (lambda s: dataclasses.replace(s, source_id=""), "source_id"),
            (lambda s: dataclasses.replace(s, poll_interval_seconds=0.0), "poll_interval_seconds"),
        ],
    )
    def test_when_field_emptied_then_rejected(
        self,
        valid: IngestionSettings,
        change: Callable[[IngestionSettings], IngestionSettings],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            change(valid)


class TestGivenSystemClock:
    """The production clock."""

    def test_when_now_then_utc_millisecond_precision(self) -> None:
        now = bootstrap.SystemClock().now()
        assert now.tzinfo is UTC
        assert now.microsecond % 1000 == 0


class TestGivenWorkerLoop:
    """`run_worker` over a counting service and a fake client, settings from `environment`."""

    @pytest.fixture
    def wire(self, environment: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Wire:
        def start(
            failing: frozenset[int] = frozenset(), poll_interval: str | None = "2.5"
        ) -> tuple[CountingService, FakeMongoClient]:
            if poll_interval is None:
                del environment[settings.POLL_INTERVAL_SECONDS_ENV]
            else:
                environment[settings.POLL_INTERVAL_SECONDS_ENV] = poll_interval
            loaded = bootstrap.load_settings(environment)
            service = CountingService(failing)
            client = FakeMongoClient(loaded.registry_uri)
            monkeypatch.setattr(bootstrap, "load_settings", lambda: loaded)
            monkeypatch.setattr(bootstrap, "connect_mongo", lambda settings: client)
            monkeypatch.setattr(bootstrap, "create_service", lambda settings, client: service)
            return service, client

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
        service, client = wire()

        bootstrap.run_worker(once=True)

        assert service.passes == 1
        assert client.entered and client.closed

    def test_when_pass_fails_once_then_error_propagates_and_client_closed(self, wire: Wire) -> None:
        _, client = wire(failing=frozenset({1}))

        with pytest.raises(OSError, match="pass 1 failed"):
            bootstrap.run_worker(once=True)

        assert client.closed

    def test_when_pass_fails_in_loop_then_logged_and_loop_continues(
        self, wire: Wire, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _ = wire(failing=frozenset({2}))
        _, sleep = self.interrupt_after(3)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert service.passes == 3
        assert "pass 2 failed" in caplog.text

    def test_when_sleep_interrupts_then_one_pass_per_wakeup_and_client_closed(
        self, wire: Wire
    ) -> None:
        service, client = wire()
        slept, sleep = self.interrupt_after(3)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert service.passes == 3
        assert slept == [2.5, 2.5, 2.5]
        assert client.closed

    def test_when_passes_keep_failing_then_pause_doubles_until_one_succeeds(
        self, wire: Wire, caplog: pytest.LogCaptureFixture
    ) -> None:
        wire(failing=frozenset({1, 2, 3}))
        slept, sleep = self.interrupt_after(5)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert slept == [2.5, 5.0, 10.0, 2.5, 2.5]
        assert "retrying in 10.0 s" in caplog.text

    def test_when_backoff_reaches_cap_then_stays_there(self, wire: Wire) -> None:
        wire(failing=frozenset({1, 2, 3}), poll_interval="200")
        slept, sleep = self.interrupt_after(3)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert slept == [200.0, 300.0, 300.0]

    def test_when_interval_above_cap_then_interval_kept(self, wire: Wire) -> None:
        wire(failing=frozenset({1, 2}), poll_interval="400")
        slept, sleep = self.interrupt_after(2)

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert slept == [400.0, 400.0]

    def test_when_looping_without_poll_interval_then_error_before_connecting(
        self, wire: Wire
    ) -> None:
        service, client = wire(poll_interval=None)

        with pytest.raises(RuntimeError, match=settings.POLL_INTERVAL_SECONDS_ENV):
            bootstrap.run_worker(once=False)

        assert service.passes == 0
        assert not client.entered

    def test_when_once_without_poll_interval_then_runs(self, wire: Wire) -> None:
        service, _ = wire(poll_interval=None)

        bootstrap.run_worker(once=True)

        assert service.passes == 1


class TestGivenUnreachableServer:
    """Settings pointing the registry at a server that refuses the connectivity check."""

    @pytest.fixture
    def unreachable(self, environment: dict[str, str]) -> IngestionSettings:
        environment[settings.REGISTRY_URI_ENV] = "mongodb://app:s3cret@127.0.0.1:1"
        return bootstrap.load_settings(environment)

    def test_when_connecting_then_runtime_error_without_password(
        self, unreachable: IngestionSettings
    ) -> None:
        created: list[FakeMongoClient] = []

        def refusing(uri: str, **options: object) -> FakeMongoClient:
            created.append(FakeMongoClient(uri, refuse=True, **options))
            return created[-1]

        with pytest.raises(RuntimeError, match="Could not reach MongoDB") as exc_info:
            bootstrap.connect_mongo(
                unreachable, client_factory=cast("bootstrap.ClientFactory", refusing)
            )

        assert "s3cret" not in str(exc_info.value)
        assert "mongodb://***@127.0.0.1:1" in str(exc_info.value)
        (client,) = created
        assert client.uri == "mongodb://app:s3cret@127.0.0.1:1"
        assert client.options == {"tz_aware": True, "serverSelectionTimeoutMS": 2000}

    def test_when_uri_malformed_then_runtime_error(self, environment: dict[str, str]) -> None:
        environment[settings.REGISTRY_URI_ENV] = "mongodb://127.0.0.1:notaport"

        with pytest.raises(RuntimeError, match="Could not reach MongoDB"):
            bootstrap.connect_mongo(bootstrap.load_settings(environment))
