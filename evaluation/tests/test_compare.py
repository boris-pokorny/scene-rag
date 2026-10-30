"""The compare command end to end over a directory Eval Store, and the store's read side."""

import json
from pathlib import Path
from typing import Any

import pytest

from scene_rag.evaluation import bootstrap
from scene_rag.evaluation.config import RunConfig
from scene_rag.evaluation.eval_store import EvalStoreError, JsonEvalStore
from scene_rag.evaluation.models import Verdict
from scene_rag.evaluation.settings import EVAL_STORE_ROOT_ENV
from tests.support.fakes import T0

SAMPLE = (
    Path(__file__).resolve().parents[1] / "contracts" / "internal" / "samples" / "eval-run.json"
)


def stored_run(root: Path, run_id: str, **changes: object) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(SAMPLE.read_text())
    document["run_id"] = run_id
    document.update(changes)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / f"{run_id}.json").write_text(json.dumps(document))
    return document


def compare(root: Path, candidate: str, baseline: str | None) -> Verdict:
    return bootstrap.compare(
        candidate, baseline, {EVAL_STORE_ROOT_ENV: str(root)}, clock=lambda: T0
    )


class TestGivenTwoStoredRuns:
    """Candidate and baseline in the Eval Store."""

    def test_when_compared_then_table_printed_and_verdict_stored(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stored_run(tmp_path, "run-000000000001")
        stored_run(tmp_path, "run-000000000002")

        verdict = compare(tmp_path, "run-000000000002", "run-000000000001")

        assert verdict.passed
        out = capsys.readouterr().out
        assert "tag:visual" in out and "+0.0000" in out and "verdict: passed" in out
        written = json.loads((tmp_path / "verdicts" / "run-000000000002.json").read_text())
        assert written["verdict"] == "passed"
        assert written["baseline_run_id"] == "run-000000000001"
        assert written["decided_at"] == T0.isoformat()

    def test_when_candidate_leaks_then_failed_and_stored(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stored_run(tmp_path, "run-000000000001")
        leaking = stored_run(tmp_path, "run-000000000002")
        leaking["aggregates"]["overall"]["answer/leakage"] = 1.0
        (tmp_path / "runs" / "run-000000000002.json").write_text(json.dumps(leaking))

        verdict = compare(tmp_path, "run-000000000002", "run-000000000001")

        assert not verdict.passed
        assert "FAIL answer/leakage: 1" in capsys.readouterr().out
        written = json.loads((tmp_path / "verdicts" / "run-000000000002.json").read_text())
        assert written["verdict"] == "failed"

    def test_when_no_baseline_then_no_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stored_run(tmp_path, "run-000000000002")

        assert compare(tmp_path, "run-000000000002", None).passed
        assert "scope" not in capsys.readouterr().out

    def test_when_datasets_differ_then_runtime_error(self, tmp_path: Path) -> None:
        stored_run(tmp_path, "run-000000000001")
        stored_run(
            tmp_path,
            "run-000000000002",
            dataset={"name": "example", "version": "2.0.0", "sha256": "0" * 64},
        )

        with pytest.raises(RuntimeError, match="same dataset version"):
            compare(tmp_path, "run-000000000002", "run-000000000001")


class TestGivenEnvironment:
    """The Eval Store's directory is required."""

    def test_when_root_missing_then_named(self) -> None:
        with pytest.raises(RuntimeError, match=EVAL_STORE_ROOT_ENV):
            bootstrap.load_eval_store_root({})


class TestGivenStoreReadSide:
    """Malformed ids, missing and undecodable runs."""

    @pytest.mark.parametrize("run_id", ["../etc/passwd", "run-XYZ", "run-0000000000001"])
    def test_when_id_malformed_then_rejected(self, tmp_path: Path, run_id: str) -> None:
        with pytest.raises(EvalStoreError, match="not a run id"):
            JsonEvalStore(tmp_path).read(run_id)

    def test_when_missing_then_named(self, tmp_path: Path) -> None:
        with pytest.raises(EvalStoreError, match="no run run-000000000009"):
            JsonEvalStore(tmp_path).read("run-000000000009")

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("{nope", "cannot read"),
            ("[]", "not a JSON object"),
            ('{"schema_version": 1}', "not an eval run"),
            ('{"schema_version": 9}', "unsupported schema_version"),
        ],
    )
    def test_when_undecodable_then_rejected(
        self, tmp_path: Path, content: str, message: str
    ) -> None:
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "run-000000000001.json").write_text(content)

        with pytest.raises(EvalStoreError, match=message):
            JsonEvalStore(tmp_path).read("run-000000000001")

    def test_when_use_case_unknown_then_rejected(self, tmp_path: Path) -> None:
        stored_run(
            tmp_path, "run-000000000001", config={"ks": [5], "top_k": 5, "use_cases": ["rerank"]}
        )

        with pytest.raises(EvalStoreError, match="unknown use case"):
            JsonEvalStore(tmp_path).read("run-000000000001")

    def test_when_sample_read_then_summary_decoded(self, tmp_path: Path) -> None:
        stored_run(tmp_path, "run-0b7c1d2e3f40")

        run = JsonEvalStore(tmp_path).read("run-0b7c1d2e3f40")

        assert run.dataset.version == "1.0.0" and run.errors == 0
        assert run.aggregates.overall["search/recall@5"] == 1.0
        assert run.served is not None and run.served.chunking_fingerprint is None
        assert run.config == RunConfig() and len(run.cases) == 3
