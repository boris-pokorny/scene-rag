import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scene_rag.extraction.ingestion.config import IngestionConfig
from scene_rag.extraction.ingestion.sources import (
    FilesystemMediaSource,
    asset_id_for,
    to_millisecond_utc,
)

ENTITLEMENTS = frozenset({"public"})
STAMP = 1_761_555_330.123_456_7


def make_source(root: Path, config: IngestionConfig | None = None) -> FilesystemMediaSource:
    return FilesystemMediaSource(root, "local", ENTITLEMENTS, config or IngestionConfig())


class TestGivenDirectory:
    """A directory holding two videos, a nested video, a text file, a stream and a decoy dir.

    ``a.MOV`` carries a known modification time with sub-millisecond digits.
    """

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "b.mp4").write_bytes(b"bbbb")
        (tmp_path / "a.MOV").write_bytes(b"aa")
        os.utime(tmp_path / "a.MOV", (STAMP, STAMP))
        (tmp_path / "notes.txt").write_text("skip")
        (tmp_path / "stream.ts").write_bytes(b"t")
        (tmp_path / "video.mp4").mkdir()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.mp4").write_bytes(b"d")
        return tmp_path

    def test_when_listed_then_accepted_files_in_path_order(self, root: Path) -> None:
        items = list(make_source(root).list())

        assert [item.filename for item in items] == ["a.MOV", "b.mp4", "deep.mp4"]
        first = items[0]
        assert first.media_type == "video/quicktime"
        assert first.size_bytes == 2
        assert first.uri == (root / "a.MOV").resolve().as_uri()
        assert first.entitlements == ENTITLEMENTS
        with first.open() as stream:
            assert stream.read() == b"aa"

    def test_when_file_nested_then_asset_id_from_source_and_relative_path(self, root: Path) -> None:
        deep = [item for item in make_source(root).list() if item.filename == "deep.mp4"][0]

        assert deep.asset_id == asset_id_for("local", "sub/deep.mp4")
        assert deep.asset_id.startswith("asset-") and len(deep.asset_id) == 22
        assert asset_id_for("other", "sub/deep.mp4") != deep.asset_id
        expected = hashlib.sha256(b"local\nsub/deep.mp4").hexdigest()[:16]
        assert deep.asset_id == f"asset-{expected}"

    def test_when_not_recursive_then_top_level_only(self, root: Path) -> None:
        items = make_source(root, IngestionConfig(recursive=False)).list()

        assert [item.filename for item in items] == ["a.MOV", "b.mp4"]

    def test_when_media_types_customised_then_only_those_extensions(self, root: Path) -> None:
        items = make_source(root, IngestionConfig(media_types={".ts": "video/mp2t"})).list()

        assert [(i.filename, i.media_type) for i in items] == [("stream.ts", "video/mp2t")]

    def test_when_file_vanishes_mid_listing_then_skipped(
        self, root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        original = Path.stat

        def stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
            if self.name == "b.mp4":
                raise FileNotFoundError(self)
            return original(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "stat", stat)

        items = list(make_source(root).list())

        assert [item.filename for item in items] == ["a.MOV", "deep.mp4"]
        assert "skipped" in caplog.text and "b.mp4" in caplog.text

    def test_when_root_missing_then_raises_instead_of_empty_listing(self, tmp_path: Path) -> None:
        source = make_source(tmp_path / "unmounted")

        with pytest.raises(FileNotFoundError, match="source root"):
            next(source.list())

    def test_when_listed_then_modified_at_truncated_to_milliseconds(self, root: Path) -> None:
        first = next(make_source(root).list())

        assert first.modified_at.tzinfo is UTC
        assert first.modified_at == datetime.fromtimestamp(STAMP, UTC).replace(microsecond=123_000)
        assert to_millisecond_utc(STAMP) == first.modified_at
