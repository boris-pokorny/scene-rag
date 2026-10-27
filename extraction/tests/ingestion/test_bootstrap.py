import dataclasses
from collections.abc import Callable
from datetime import UTC
from pathlib import Path

import pytest

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

    @pytest.mark.parametrize(
        "missing",
        [
            settings.REGISTRY_URI_ENV,
            settings.REGISTRY_DATABASE_ENV,
            settings.SOURCE_ROOT_ENV,
            settings.SOURCE_ID_ENV,
            settings.SOURCE_ENTITLEMENTS_ENV,
            settings.POLL_INTERVAL_SECONDS_ENV,
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
            (settings.SOURCE_ROOT_ENV, "missing-directory", "source_root"),
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
    """`run_worker` with settings loaded from `environment` and a counting service."""

    @pytest.fixture
    def service(
        self, environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> CountingService:
        loaded = bootstrap.load_settings(environment)
        service = CountingService(failing=frozenset({2}))
        monkeypatch.setattr(bootstrap, "load_settings", lambda: loaded)
        monkeypatch.setattr(bootstrap, "create_service", lambda settings: service)
        return service

    def test_when_pass_fails_in_loop_then_logged_and_loop_continues(
        self, service: CountingService, caplog: pytest.LogCaptureFixture
    ) -> None:
        def sleep(seconds: float) -> None:
            if service.passes == 3:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert service.passes == 3
        assert "pass 2 failed" in caplog.text

    def test_when_pass_fails_once_then_error_propagates(self, service: CountingService) -> None:
        service.passes = 1  # the next pass is the failing one

        with pytest.raises(OSError, match="pass 2 failed"):
            bootstrap.run_worker(once=True)

    def test_when_sleep_interrupts_then_one_pass_per_wakeup(self, service: CountingService) -> None:
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            if len(slept) == 3:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_worker(once=False, sleep=sleep)

        assert service.passes == 3
        assert slept == [2.5, 2.5, 2.5]


class TestGivenUnreachableServer:
    """Settings pointing the registry at a closed port."""

    @pytest.fixture
    def unreachable(self, environment: dict[str, str]) -> IngestionSettings:
        environment[settings.REGISTRY_URI_ENV] = "mongodb://app:s3cret@127.0.0.1:1"
        return bootstrap.load_settings(environment)

    def test_when_connecting_then_runtime_error_without_password(
        self, unreachable: IngestionSettings
    ) -> None:
        with pytest.raises(RuntimeError, match="Could not reach MongoDB") as exc_info:
            bootstrap.connect_mongo(unreachable, timeout_ms=100)

        assert "s3cret" not in str(exc_info.value)
        assert "mongodb://***@127.0.0.1:1" in str(exc_info.value)

    def test_when_uri_malformed_then_runtime_error(self, environment: dict[str, str]) -> None:
        environment[settings.REGISTRY_URI_ENV] = "mongodb://127.0.0.1:notaport"

        with pytest.raises(RuntimeError, match="Could not reach MongoDB"):
            bootstrap.connect_mongo(bootstrap.load_settings(environment), timeout_ms=100)
