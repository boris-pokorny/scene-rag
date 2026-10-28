from datetime import UTC, datetime

from scene_rag.extraction.segmentation.models import (
    SegmentationReport,
    scene_set_version,
    truncate_to_millisecond,
)
from tests.support.segmentation_fakes import make_segments, make_view, sha256_version


class TestGivenSceneSetVersion:
    """The data version derived from an asset version and a fingerprint."""

    def test_when_same_inputs_then_same_version(self) -> None:
        asset = sha256_version(b"x")

        assert scene_set_version(asset, "abc", "sha256") == scene_set_version(
            asset, "abc", "sha256"
        )
        assert scene_set_version(asset, "abc", "sha256").startswith("sha256:")

    def test_when_fingerprint_or_asset_differs_then_version_differs(self) -> None:
        asset = sha256_version(b"x")

        assert scene_set_version(asset, "abc", "sha256") != scene_set_version(
            asset, "abd", "sha256"
        )
        assert scene_set_version(asset, "abc", "sha256") != scene_set_version(
            sha256_version(b"y"), "abc", "sha256"
        )
        assert scene_set_version(asset, "abc", "sha1").startswith("sha1:")


class TestGivenView:
    """An asset version as the step sees it."""

    def test_when_retired_or_sourceless_then_not_live(self) -> None:
        assert make_view().is_live
        assert not make_view(retired_at=datetime(2026, 1, 1, tzinfo=UTC)).is_live
        assert (
            not make_view()
            .__class__(version="v", media_type="video/mp4", size_bytes=1, sources=())
            .is_live
        )


class TestGivenSegment:
    """A time span."""

    def test_when_asked_for_duration_then_difference_of_bounds(self) -> None:
        assert [s.duration_seconds for s in make_segments()] == [4.0, 4.0, 4.5]


class TestGivenTimestamp:
    """Datetimes as the clock produces them."""

    def test_when_truncated_then_microseconds_dropped(self) -> None:
        moment = datetime(2026, 10, 27, 9, 0, 0, 123_456, tzinfo=UTC)

        assert truncate_to_millisecond(moment) == moment.replace(microsecond=123_000)


class TestGivenReport:
    """The pass counters."""

    def test_when_rendered_then_key_value_pairs(self) -> None:
        assert str(SegmentationReport(segmented=2, failed=1)) == (
            "segmented=2 duplicate=0 dropped=0 deferred=0 failed=1"
        )

    def test_when_counted_then_one_per_outcome_whatever_the_order(self) -> None:
        report = SegmentationReport.counting(
            ["failed", "segmented", "deferred", "segmented", "dropped", "duplicate", "segmented"]
        )

        assert report == SegmentationReport(
            segmented=3, duplicate=1, dropped=1, deferred=1, failed=1
        )

    def test_when_counted_over_nothing_then_zero(self) -> None:
        assert SegmentationReport.counting(iter(())) == SegmentationReport()
