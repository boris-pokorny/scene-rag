"""The run role's composition root: settings, metric registry, wiring, the whole run."""

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from scene_rag.evaluation import bootstrap
from scene_rag.evaluation.config import RunConfig
from scene_rag.evaluation.dataset import load_dataset
from scene_rag.evaluation.eval_store import JsonEvalStore
from scene_rag.evaluation.http_transport import Json, TransportError
from scene_rag.evaluation.service import metric_key
from scene_rag.evaluation.settings import (
    EVAL_STORE_ROOT_ENV,
    QUERY_API_TIMEOUT_SECONDS_ENV,
    QUERY_API_URL_ENV,
    RunSettings,
)
from tests.test_query_api_client import sample

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "example" / "1.0.0.jsonl"


def environ(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {QUERY_API_URL_ENV: "http://q:8081", EVAL_STORE_ROOT_ENV: str(tmp_path), **extra}


class QueryApiStandIn:
    """Answers query-api's routes with retrieval's published samples."""

    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    def exchange(
        self,
        method: str,
        url: str,
        payload: Json | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Json:
        if url.endswith("/health"):
            if not self.healthy:
                raise TransportError("could not reach http://q:8081/health")
            return {"status": "ok"}
        return sample("search-response" if url.endswith("/search") else "answer-response")


class TestGivenEnvironment:
    """Settings bound from the environment."""

    def test_when_complete_then_bound_with_default_timeout(self, tmp_path: Path) -> None:
        settings = bootstrap.load_settings(environ(tmp_path))

        assert settings == RunSettings("http://q:8081", tmp_path, 300.0)

    def test_when_timeout_given_then_bound(self, tmp_path: Path) -> None:
        env = environ(tmp_path, **{QUERY_API_TIMEOUT_SECONDS_ENV: "12.5"})

        assert bootstrap.load_settings(env).query_api_timeout_seconds == 12.5

    @pytest.mark.parametrize("missing", [QUERY_API_URL_ENV, EVAL_STORE_ROOT_ENV])
    def test_when_required_missing_then_named(self, tmp_path: Path, missing: str) -> None:
        env = environ(tmp_path)
        del env[missing]

        with pytest.raises(RuntimeError, match=missing):
            bootstrap.load_settings(env)

    @pytest.mark.parametrize(
        ("value", "message"), [("soon", "must be a number"), ("-1", "invalid")]
    )
    def test_when_timeout_invalid_then_rejected(
        self, tmp_path: Path, value: str, message: str
    ) -> None:
        env = environ(tmp_path, **{QUERY_API_TIMEOUT_SECONDS_ENV: value})

        with pytest.raises(RuntimeError, match=message):
            bootstrap.load_settings(env)


class TestGivenMetricRegistry:
    """Every shipped metric, once per configured cut-off."""

    def test_when_built_then_keys_unique_and_complete(self) -> None:
        keys = [metric_key(m) for m in bootstrap.build_metrics(RunConfig(ks=(1, 5), top_k=5))]

        assert len(keys) == len(set(keys))
        assert {"search/hit@1", "search/recall@5", "search/ndcg@5", "search/mrr"} <= set(keys)
        assert {"search/leakage", "answer/leakage", "answer/latency_ms"} <= set(keys)
        assert {"answer/citation_validity", "answer/refusal_correctness"} <= set(keys)
        assert {"answer/prompt_tokens", "answer/completion_tokens"} <= set(keys)


class TestGivenQueryApi:
    """Wiring checks that query-api answers before the first case."""

    def test_when_unreachable_then_named(self, tmp_path: Path) -> None:
        settings = bootstrap.load_settings(environ(tmp_path))

        with pytest.raises(RuntimeError, match="query-api at http://q:8081 does not answer"):
            bootstrap.build_service(settings, transport=QueryApiStandIn(healthy=False))

    def test_when_example_run_then_stored_and_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings = bootstrap.load_settings(environ(tmp_path))
        service = bootstrap.build_service(settings, transport=QueryApiStandIn())

        run = service.run(load_dataset(EXAMPLE))
        print("\n".join(bootstrap.format_report(run)))

        assert re.fullmatch(r"run-[0-9a-f]{12}", run.run_id)
        stored = json.loads((tmp_path / "runs" / f"{run.run_id}.json").read_text())
        assert stored["run_id"] == run.run_id and stored["errors"] == 0
        # The sample's sources overlap q-001's span, so search finds it.
        assert stored["aggregates"]["overall"]["search/recall@5"] == 1.0
        out = capsys.readouterr().out
        assert "example@1.0.0" in out and "query_config 11d0a9e2c4f1" in out
        assert "search/recall@5" in out and "errors  0" in out


class TestGivenRun:
    """The role's entry point."""

    def test_when_dataset_invalid_then_runtime_error(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="invalid dataset"):
            bootstrap.run(tmp_path / "1.0.0.jsonl", environ(tmp_path))

    def test_when_run_then_built_run_and_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        real = bootstrap.build_service

        def build(settings: RunSettings) -> object:
            return real(settings, transport=QueryApiStandIn(), writer=JsonEvalStore(tmp_path))

        monkeypatch.setattr(bootstrap, "build_service", build)

        run = bootstrap.run(EXAMPLE, environ(tmp_path))

        assert (tmp_path / "runs" / f"{run.run_id}.json").exists()
        assert run.run_id in capsys.readouterr().out

    def test_when_release_changes_then_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Promoting(QueryApiStandIn):
            def exchange(
                self,
                method: str,
                url: str,
                payload: Json | None,
                headers: Mapping[str, str],
                timeout: float,
            ) -> Json:
                document = super().exchange(method, url, payload, headers, timeout)
                if url.endswith("/answer"):
                    document["served"] = {**document["served"], "query_config": "999999999999"}
                return document

        real = bootstrap.build_service
        monkeypatch.setattr(
            bootstrap, "build_service", lambda settings: real(settings, transport=Promoting())
        )

        with pytest.raises(RuntimeError, match="changed what it serves"):
            bootstrap.run(EXAMPLE, environ(tmp_path))
