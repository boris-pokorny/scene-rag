import dataclasses
from pathlib import Path

import pytest

from scene_rag.extraction.ingestion.settings import IngestionSettings, redact_uri

VALID = IngestionSettings(
    registry_uri="mongodb://unused:27017",
    registry_database="scene_rag_test",
    source_root=Path("/nowhere/media"),
    source_id="local",
    source_entitlements=frozenset({"public"}),
    poll_interval_seconds=2.5,
)


class TestGivenRegistryUri:
    """Connection strings as they may appear in the environment."""

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("mongodb://localhost:27017", "mongodb://localhost:27017"),
            ("mongodb://app:s3cret@mongo:27017/db", "mongodb://***@mongo:27017/db"),
            (
                "mongodb+srv://app:p%40ss@cluster.example.net/",
                "mongodb+srv://***@cluster.example.net/",
            ),
            ("mongodb://app@mongo:27017", "mongodb://***@mongo:27017"),
        ],
    )
    def test_when_redacted_then_userinfo_hidden(self, uri: str, expected: str) -> None:
        assert redact_uri(uri) == expected


class TestGivenSettingsValues:
    """The value object on its own: shape checks only, no filesystem."""

    def test_when_root_does_not_exist_then_accepted(self) -> None:
        assert not VALID.source_root.exists()
        assert VALID.redacted_registry_uri == "mongodb://unused:27017"

    def test_when_poll_interval_omitted_then_none(self) -> None:
        single_pass = dataclasses.replace(VALID, poll_interval_seconds=None)

        assert single_pass.poll_interval_seconds is None

    @pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
    def test_when_poll_interval_not_positive_finite_then_rejected(self, interval: float) -> None:
        with pytest.raises(ValueError, match="poll_interval_seconds"):
            dataclasses.replace(VALID, poll_interval_seconds=interval)
