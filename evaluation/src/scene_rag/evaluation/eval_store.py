"""The Eval Store over a directory: ``<root>/runs/`` and ``<root>/verdicts/``.

A run's document is `contracts/internal/eval-run.schema.json`; it is read back by `compare`,
possibly long after the code that wrote it changed, so the decoder checks the schema version.
A verdict's document is `contracts/published/eval-verdict.schema.json`, the contract a release
step reads. Every document is written to a temporary file and renamed, so a reader never sees
half of one.
"""

import dataclasses
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from scene_rag.evaluation.config import USE_CASES, RunConfig, UseCase
from scene_rag.evaluation.models import (
    Aggregates,
    CaseResult,
    DatasetRef,
    EvalRun,
    Served,
    Verdict,
)

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]

RUN_ID_PATTERN: Final = re.compile(r"^run-[0-9a-f]{12}$")
SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({1})


class EvalStoreError(RuntimeError):
    """A run id is malformed, or the run is missing or undecodable."""


def _served(served: Served | None) -> Json | None:
    return None if served is None else dataclasses.asdict(served)


def encode_run(run: EvalRun) -> Json:
    """Encode ``run`` as its stored document."""
    return {
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat(),
        "dataset": dataclasses.asdict(run.dataset),
        "served": _served(run.served),
        "config": {
            "ks": list(run.config.ks),
            "top_k": run.config.top_k,
            "use_cases": list(run.config.use_cases),
        },
        "cases": [
            {
                "id": case.case_id,
                "tags": list(case.tags),
                "metrics": dict(sorted(case.metrics.items())),
                "errors": list(case.errors),
            }
            for case in run.cases
        ],
        "aggregates": {
            "overall": dict(run.aggregates.overall),
            "by_tag": {tag: dict(values) for tag, values in run.aggregates.by_tag.items()},
        },
        "errors": run.errors,
    }


def _floats(document: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in document.items()}


def _use_case(value: object) -> UseCase:
    for use_case in USE_CASES:
        if value == use_case:
            return use_case
    raise ValueError(f"unknown use case {value!r}")


def _served_from_json(document: Mapping[str, Any] | None) -> Served | None:
    if document is None:
        return None
    chunking = document["chunking_fingerprint"]
    return Served(
        query_config=str(document["query_config"]),
        embedding_fingerprint=str(document["embedding_fingerprint"]),
        chunking_fingerprint=None if chunking is None else str(chunking),
        collection=str(document["collection"]),
    )


def decode_run(document: Mapping[str, Any]) -> EvalRun:
    """Decode a stored run; the inverse of `encode_run`.

    Raises:
        ValueError: When the document is not a supported eval run.
    """
    try:
        if document["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version {document['schema_version']!r}")
        dataset, config, aggregates = (
            document["dataset"],
            document["config"],
            document["aggregates"],
        )
        return EvalRun(
            run_id=str(document["run_id"]),
            started_at=datetime.fromisoformat(document["started_at"]),
            finished_at=datetime.fromisoformat(document["finished_at"]),
            dataset=DatasetRef(
                name=str(dataset["name"]),
                version=str(dataset["version"]),
                sha256=str(dataset["sha256"]),
            ),
            served=_served_from_json(document["served"]),
            config=RunConfig(
                ks=tuple(int(k) for k in config["ks"]),
                top_k=int(config["top_k"]),
                use_cases=tuple(_use_case(u) for u in config["use_cases"]),
            ),
            cases=tuple(
                CaseResult(
                    case_id=str(case["id"]),
                    tags=tuple(str(tag) for tag in case["tags"]),
                    metrics=_floats(case["metrics"]),
                    errors=tuple(str(error) for error in case["errors"]),
                )
                for case in document["cases"]
            ),
            aggregates=Aggregates(
                overall=_floats(aggregates["overall"]),
                by_tag={str(tag): _floats(v) for tag, v in aggregates["by_tag"].items()},
            ),
            errors=int(document["errors"]),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"not an eval run: {exc!r}") from exc


def encode_verdict(verdict: Verdict) -> Json:
    """Encode ``verdict`` as its published document."""
    return {
        "schema_version": verdict.schema_version,
        "run_id": verdict.run_id,
        "baseline_run_id": verdict.baseline_run_id,
        "dataset": dataclasses.asdict(verdict.dataset),
        "served": _served(verdict.served),
        "verdict": "passed" if verdict.passed else "failed",
        "checks": [dataclasses.asdict(check) for check in verdict.checks],
        "gate": {
            "primary_metrics": list(verdict.gate.primary_metrics),
            "tolerance": verdict.gate.tolerance,
            "zero_metrics": list(verdict.gate.zero_metrics),
        },
        "decided_at": verdict.decided_at.isoformat(),
    }


def _write(path: Path, document: Json) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)
    return str(path)


class JsonEvalStore:
    """The Eval Store over a directory; `EvalRunWriter` for the run use case."""

    def __init__(self, root: Path) -> None:
        """Bind the Eval Store's directory; subdirectories are created on first write."""
        self._runs = root / "runs"
        self._verdicts = root / "verdicts"

    def write(self, run: EvalRun) -> str:
        """Write ``run`` atomically and return its path."""
        return _write(self._runs / f"{run.run_id}.json", encode_run(run))

    def read(self, run_id: str) -> EvalRun:
        """Return the stored run ``run_id``.

        Raises:
            EvalStoreError: When the id is malformed or the run missing or undecodable.
        """
        if not RUN_ID_PATTERN.match(run_id):
            raise EvalStoreError(f"{run_id!r} is not a run id (run-<12 hex digits>)")
        path = self._runs / f"{run_id}.json"
        try:
            document = json.loads(path.read_text())
        except FileNotFoundError:
            raise EvalStoreError(f"no run {run_id} in {self._runs}") from None
        except (OSError, ValueError) as exc:
            raise EvalStoreError(f"cannot read {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise EvalStoreError(f"{path} is not a JSON object")
        try:
            return decode_run(document)
        except ValueError as exc:
            raise EvalStoreError(f"{path}: {exc}") from exc

    def write_verdict(self, verdict: Verdict) -> str:
        """Write ``verdict`` atomically, replacing an earlier one for the run; return its path."""
        return _write(self._verdicts / f"{verdict.run_id}.json", encode_verdict(verdict))
