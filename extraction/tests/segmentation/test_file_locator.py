import os
from datetime import timedelta
from pathlib import Path

import pytest

from scene_rag.extraction.segmentation.file_locator import FileUriLocator, to_millisecond_utc
from scene_rag.extraction.segmentation.protocols import MediaUnavailableError
from tests.support.segmentation_fakes import make_source, make_view

STAMP = 1_761_555_330.123_456_7


class TestGivenRecordedFile:
    """A file on disk and a source reference recorded by ingestion when it hashed it."""

    @pytest.fixture
    def file(self, tmp_path: Path) -> Path:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"abcd")
        os.utime(path, (STAMP, STAMP))
        return path

    def test_when_unchanged_then_path_returned(self, file: Path) -> None:
        view = make_view(
            "sha256:x",
            make_source(uri=file.as_uri(), size_bytes=4, modified_at=to_millisecond_utc(STAMP)),
        )

        assert FileUriLocator().locate(view) == file

    def test_when_first_source_unusable_then_next_tried(self, file: Path) -> None:
        good = make_source(uri=file.as_uri(), size_bytes=4, modified_at=to_millisecond_utc(STAMP))
        view = make_view(
            "sha256:x",
            make_source(uri="s3://bucket/clip.mp4"),
            make_source(uri=(file.parent / "gone.mp4").as_uri()),
            good,
        )

        assert FileUriLocator().locate(view) == file

    @pytest.mark.parametrize(
        ("size_bytes", "offset", "reason"),
        [(5, 0.0, "size changed"), (4, 1.0, "modified since")],
    )
    def test_when_file_changed_since_ingestion_then_unavailable(
        self, file: Path, size_bytes: int, offset: float, reason: str
    ) -> None:
        view = make_view(
            "sha256:x",
            make_source(
                uri=file.as_uri(),
                size_bytes=size_bytes,
                modified_at=to_millisecond_utc(STAMP) + timedelta(seconds=offset),
            ),
        )

        with pytest.raises(MediaUnavailableError, match=reason):
            FileUriLocator().locate(view)

    def test_when_no_source_usable_then_every_reason_listed(self, tmp_path: Path) -> None:
        view = make_view(
            "sha256:x",
            make_source(uri="https://example.net/clip.mp4"),
            make_source(uri=(tmp_path / "missing.mp4").as_uri()),
        )

        with pytest.raises(MediaUnavailableError) as exc_info:
            FileUriLocator().locate(view)

        assert "not a file URI" in str(exc_info.value)
        assert "missing.mp4" in str(exc_info.value)
