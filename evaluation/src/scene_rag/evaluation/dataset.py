"""Reader of versioned JSONL datasets: ``data/<name>/<version>.jsonl``.

One case per line, in the shape of `contracts/internal/dataset-case.schema.json`. Relevance
is labelled by asset version and time span, not by chunk, so one dataset stays valid across
chunking strategies and rebuilds. The whole file is validated before the first case is put:
a bad label would otherwise turn into a misleading score.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from scene_rag.evaluation.models import Case, Dataset, DatasetRef, Span

VERSION_PATTERN: Final = re.compile(r"^\d+\.\d+\.\d+$")
_CASE_KEYS: Final = frozenset(
    {"id", "question", "relevant", "forbidden", "reference_answer", "tags", "caller_entitlements"}
)
_SPAN_KEYS: Final = frozenset({"asset_id", "asset_version", "start_seconds", "end_seconds"})
_ENTITLEMENT: Final = re.compile(r"^[!-+\--~]+$")
"""Printable ASCII without space or comma: what survives query-api's comma-separated header."""


class DatasetError(ValueError):
    """The dataset file is misnamed or a case does not validate."""


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _seconds(document: Mapping[str, Any], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{key} must be a finite number of at least 0")
    return float(value)


def _names(document: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = document.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)


def _entitlements(document: Mapping[str, Any]) -> frozenset[str]:
    names = _names(document, "caller_entitlements")
    bad = [name for name in names if not _ENTITLEMENT.match(name)]
    if bad:
        raise ValueError(
            f"caller_entitlements must be printable ASCII without spaces or commas, not {bad}"
        )
    return frozenset(names)


def _span(document: object) -> Span:
    if not isinstance(document, dict):
        raise ValueError("a span must be an object")
    unknown = document.keys() - _SPAN_KEYS
    if unknown:
        raise ValueError(f"unknown span fields {sorted(unknown)}")
    span = Span(
        asset_id=_text(document, "asset_id"),
        asset_version=_text(document, "asset_version"),
        start_seconds=_seconds(document, "start_seconds"),
        end_seconds=_seconds(document, "end_seconds"),
    )
    if span.end_seconds <= span.start_seconds:
        raise ValueError("end_seconds must be after start_seconds")
    return span


def _spans(document: Mapping[str, Any], key: str) -> tuple[Span, ...]:
    value = document.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of spans")
    return tuple(_span(item) for item in value)


def case_from_json(document: object) -> Case:
    """Decode and validate one case.

    Raises:
        ValueError: Naming the field, when the case does not validate.
    """
    if not isinstance(document, dict):
        raise ValueError("a case must be a JSON object")
    unknown = document.keys() - _CASE_KEYS
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    if "relevant" not in document:
        raise ValueError("relevant must be given; an empty list marks an unanswerable case")
    reference = document.get("reference_answer")
    if reference is not None and not isinstance(reference, str):
        raise ValueError("reference_answer must be a string")
    return Case(
        id=_text(document, "id"),
        question=_text(document, "question"),
        relevant=_spans(document, "relevant"),
        forbidden=_spans(document, "forbidden"),
        tags=_names(document, "tags"),
        caller_entitlements=_entitlements(document),
        reference_answer=reference,
    )


def parse_cases(lines: Sequence[str]) -> tuple[Case, ...]:
    """Decode every non-blank line into a case.

    Raises:
        DatasetError: Naming the line, when one does not decode or validate, or when two
            cases share an id.
    """
    cases: list[Case] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = case_from_json(json.loads(line))
        except ValueError as exc:
            raise DatasetError(f"line {number}: {exc}") from exc
        if case.id in seen:
            raise DatasetError(f"line {number}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError("the dataset holds no case")
    return tuple(cases)


def load_dataset(path: Path) -> Dataset:
    """Read and validate a dataset file.

    Args:
        path: ``<name>/<version>.jsonl``; the directory names the dataset, the file stem is
            its semantic version.

    Returns:
        The dataset, with its name, version and file digest.

    Raises:
        DatasetError: When the file is misnamed, unreadable or does not validate.
    """
    if path.suffix != ".jsonl" or not VERSION_PATTERN.match(path.stem):
        raise DatasetError(f"{path} must be named <name>/<major.minor.patch>.jsonl")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    try:
        text = raw.decode()
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path} is not UTF-8") from exc
    ref = DatasetRef(
        name=path.resolve().parent.name,
        version=path.stem,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    return Dataset(ref=ref, cases=parse_cases(text.splitlines()))
