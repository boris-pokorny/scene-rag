"""The JSONL dataset reader and its validation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scene_rag.evaluation.dataset import DatasetError, case_from_json, load_dataset, parse_cases

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "example" / "1.0.0.jsonl"
SPAN = {"asset_id": "a", "asset_version": "sha256:ab", "start_seconds": 1, "end_seconds": 2.5}


def case(**overrides: object) -> dict[str, Any]:
    document: dict[str, Any] = {"id": "q-1", "question": "What?", "relevant": [SPAN]}
    document.update(overrides)
    return document


class TestGivenShippedExample:
    """The example dataset in the repository."""

    def test_when_loaded_then_named_versioned_and_hashed(self) -> None:
        dataset = load_dataset(EXAMPLE)

        assert (dataset.ref.name, dataset.ref.version) == ("example", "1.0.0")
        assert dataset.ref.sha256 == hashlib.sha256(EXAMPLE.read_bytes()).hexdigest()
        assert [c.id for c in dataset.cases] == ["q-001", "q-002", "q-003"]
        assert [c.answerable for c in dataset.cases] == [True, False, False]
        assert dataset.cases[2].forbidden and dataset.cases[0].caller_entitlements == {"public"}


class TestGivenCase:
    """One line decoded and validated."""

    def test_when_minimal_then_defaults(self) -> None:
        decoded = case_from_json(case())

        assert decoded.forbidden == () and decoded.tags == ()
        assert decoded.caller_entitlements == frozenset() and decoded.reference_answer is None
        assert decoded.relevant[0].start_seconds == 1.0

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"id": ""}, "id"),
            ({"question": 3}, "question"),
            ({"extra": 1}, "unknown fields"),
            ({"relevant": "x"}, "relevant"),
            ({"relevant": [1]}, "span must be an object"),
            ({"relevant": [{**SPAN, "start_seconds": True}]}, "start_seconds"),
            ({"relevant": [{**SPAN, "end_seconds": -1}]}, "end_seconds"),
            ({"relevant": [{**SPAN, "end_seconds": 1}]}, "after start_seconds"),
            ({"relevant": [{**SPAN, "chunk": 1}]}, "unknown span fields"),
            ({"tags": ["", "x"]}, "tags"),
            ({"reference_answer": 1}, "reference_answer"),
            ({"caller_entitlements": ["a,b"]}, "caller_entitlements"),
            ({"caller_entitlements": [" public"]}, "caller_entitlements"),
            ({"caller_entitlements": ["veřejné"]}, "caller_entitlements"),
        ],
    )
    def test_when_invalid_then_field_named(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            case_from_json(case(**overrides))

    def test_when_relevant_missing_then_rejected(self) -> None:
        with pytest.raises(ValueError, match="relevant must be given"):
            case_from_json({"id": "q", "question": "?"})

    def test_when_not_object_then_rejected(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            case_from_json([])


class TestGivenLines:
    """A whole file's lines."""

    def test_when_blank_lines_then_skipped(self) -> None:
        assert len(parse_cases(["", json.dumps(case()), "  "])) == 1

    def test_when_line_invalid_then_line_named(self) -> None:
        with pytest.raises(DatasetError, match="line 2"):
            parse_cases([json.dumps(case()), "{not json"])

    def test_when_ids_repeat_then_rejected(self) -> None:
        with pytest.raises(DatasetError, match="duplicate case id"):
            parse_cases([json.dumps(case()), json.dumps(case())])

    def test_when_empty_then_rejected(self) -> None:
        with pytest.raises(DatasetError, match="no case"):
            parse_cases([])


class TestGivenFile:
    """Naming and reading."""

    @pytest.mark.parametrize("name", ["latest.jsonl", "1.0.jsonl", "1.0.0.json"])
    def test_when_misnamed_then_rejected(self, tmp_path: Path, name: str) -> None:
        path = tmp_path / name
        path.write_text(json.dumps(case()))

        with pytest.raises(DatasetError, match="must be named"):
            load_dataset(path)

    def test_when_missing_then_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="cannot read"):
            load_dataset(tmp_path / "1.0.0.jsonl")

    def test_when_not_utf8_then_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "1.0.0.jsonl"
        path.write_bytes(b"\xff\xfe")

        with pytest.raises(DatasetError, match="UTF-8"):
            load_dataset(path)
