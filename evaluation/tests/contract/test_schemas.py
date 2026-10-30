"""The service's JSON Schemas, their samples, and the documents the roles write."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scene_rag.evaluation.config import GateConfig
from scene_rag.evaluation.dataset import load_dataset
from scene_rag.evaluation.eval_store import decode_run, encode_run, encode_verdict
from scene_rag.evaluation.gate import decide
from scene_rag.evaluation.models import Response
from tests.support.fakes import SERVED, T0, failing, make_case, make_dataset, source, span
from tests.test_service import make_service

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
DATASETS = Path(__file__).resolve().parents[2] / "data"
SCHEMAS = {
    "dataset-case": CONTRACTS / "internal",
    "eval-run": CONTRACTS / "internal",
    "eval-verdict": CONTRACTS / "published",
}


def validator(name: str) -> jsonschema.Draft202012Validator:
    schema = json.loads((SCHEMAS[name] / f"{name}.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


def load_sample(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((SCHEMAS[name] / "samples" / f"{name}.json").read_text())
    return document


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_when_sample_validated_then_conforms(name: str) -> None:
    validator(name).validate(load_sample(name))


@pytest.mark.parametrize("path", sorted(DATASETS.rglob("*.jsonl")), ids=lambda p: p.name)
def test_when_shipped_dataset_validated_then_every_line_conforms(path: Path) -> None:
    check = validator("dataset-case")
    for line in path.read_text().splitlines():
        if line.strip():
            check.validate(json.loads(line))
    load_dataset(path)


def test_when_run_written_then_conforms_and_reads_back_unchanged() -> None:
    case = make_case("q-1", relevant=(span(10, 20),), tags=("visual",))
    service, _, _ = make_service(
        search=lambda q: Response((source(12, 14),), SERVED), answer=failing()
    )

    run = service.run(make_dataset(case))
    document = encode_run(run)

    validator("eval-run").validate(document)
    assert decode_run(json.loads(json.dumps(document))) == run
    assert run.aggregates.overall["search/recall@5"] == 1.0
    assert document["cases"][0]["errors"] == ["answer: HTTP 503"]


def test_when_run_without_any_success_then_conforms() -> None:
    service, _, _ = make_service(search=failing(), answer=failing())

    run = service.run(make_dataset(make_case()))

    validator("eval-run").validate(encode_run(run))
    assert decode_run(encode_run(run)) == run and run.served is None


def test_when_verdict_written_then_conforms() -> None:
    run = decode_run(load_sample("eval-run"))

    for baseline in (run, None):
        validator("eval-verdict").validate(encode_verdict(decide(run, baseline, GateConfig(), T0)))
