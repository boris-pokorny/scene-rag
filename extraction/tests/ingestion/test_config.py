from collections.abc import Callable

import pytest

from scene_rag.extraction.ingestion.config import IngestionConfig
from scene_rag.extraction.ingestion.fingerprint import compute_fingerprint


class TestGivenConfig:
    """The step's hyperparameters and their fingerprint."""

    def test_when_defaults_then_common_video_extensions_accepted(self) -> None:
        config = IngestionConfig()
        assert config.media_types[".mp4"] == "video/mp4"
        assert config.recursive is True
        assert config.hash_algorithm == "sha256"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: IngestionConfig(media_types={}),
            lambda: IngestionConfig(media_types={"mp4": "video/mp4"}),
            lambda: IngestionConfig(media_types={".MP4": "video/mp4"}),
            lambda: IngestionConfig(media_types={".mp4": "video"}),
            lambda: IngestionConfig(hash_algorithm="no-such-hash"),
            lambda: IngestionConfig(min_size_bytes=-1),
        ],
    )
    def test_when_value_invalid_then_rejected(self, build: Callable[[], IngestionConfig]) -> None:
        with pytest.raises(ValueError):
            build()

    def test_when_fingerprinted_twice_then_equal(self) -> None:
        assert compute_fingerprint(IngestionConfig()) == compute_fingerprint(IngestionConfig())
        assert len(compute_fingerprint(IngestionConfig())) == 12

    def test_when_hyperparameter_changes_then_fingerprint_differs(self) -> None:
        assert compute_fingerprint(IngestionConfig()) != compute_fingerprint(
            IngestionConfig(recursive=False)
        )
        assert compute_fingerprint(IngestionConfig()) != compute_fingerprint(
            IngestionConfig(media_types={".mp4": "video/mp4"})
        )
