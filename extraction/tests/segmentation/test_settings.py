import dataclasses
from pathlib import Path

import pytest

from scene_rag.extraction.segmentation.settings import SegmentationSettings, redact_uri

VALID = SegmentationSettings(
    registry_uri="mongodb://app:s3cret@unused:27017",
    registry_database="scene_rag_test",
    scene_store_uri="mongodb://scenes:27017",
    scene_store_database="scene_rag_test",
    derivative_root=Path("/nowhere/derivatives"),
    poll_interval_seconds=2.5,
)


class TestGivenSettingsValues:
    """The value object on its own: shape checks only, no filesystem."""

    def test_when_root_does_not_exist_then_accepted(self) -> None:
        assert not VALID.derivative_root.exists()
        assert VALID.ffmpeg_binary == "ffmpeg"

    def test_when_uris_redacted_then_userinfo_hidden(self) -> None:
        assert VALID.redacted_registry_uri == "mongodb://***@unused:27017"
        assert VALID.redacted_scene_store_uri == "mongodb://scenes:27017"
        assert (
            redact_uri("mongodb+srv://app:p%40ss@c.example.net/")
            == "mongodb+srv://***@c.example.net/"
        )

    def test_when_poll_interval_omitted_then_none(self) -> None:
        assert dataclasses.replace(VALID, poll_interval_seconds=None).poll_interval_seconds is None

    @pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
    def test_when_poll_interval_not_positive_finite_then_rejected(self, interval: float) -> None:
        with pytest.raises(ValueError, match="poll_interval_seconds"):
            dataclasses.replace(VALID, poll_interval_seconds=interval)

    @pytest.mark.parametrize(
        "field",
        [
            "registry_uri",
            "registry_database",
            "scene_store_uri",
            "scene_store_database",
            "ffmpeg_binary",
        ],
    )
    def test_when_field_emptied_then_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(VALID, **{field: ""})  # type: ignore[arg-type]
