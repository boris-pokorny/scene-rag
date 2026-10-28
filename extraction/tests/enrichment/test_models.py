"""Value objects of the enrichment step."""

from datetime import UTC, datetime

from scene_rag.extraction.enrichment import models
from tests.support.enrichment_fakes import make_asset_view, make_scene_set


class TestGivenEnrichmentVersion:
    """The record identifier derived from inputs and fingerprint."""

    def test_when_same_inputs_then_same_version(self) -> None:
        first = models.enrichment_version("sha256:ab", 2, "description", "0123456789ab", "sha256")

        assert first == models.enrichment_version(
            "sha256:ab", 2, "description", "0123456789ab", "sha256"
        )
        assert first.startswith("sha256:") and len(first) == len("sha256:") + 64

    def test_when_any_input_differs_then_version_differs(self) -> None:
        base = models.enrichment_version("sha256:ab", 2, "description", "0123456789ab", "sha256")

        assert (
            models.enrichment_version("sha256:ac", 2, "description", "0123456789ab", "sha256")
            != base
        )
        assert (
            models.enrichment_version("sha256:ab", 3, "description", "0123456789ab", "sha256")
            != base
        )
        assert (
            models.enrichment_version("sha256:ab", 2, "transcript", "0123456789ab", "sha256")
            != base
        )
        assert (
            models.enrichment_version("sha256:ab", 2, "description", "0123456789ac", "sha256")
            != base
        )


class TestGivenAssetVersionView:
    """Liveness of a version as the registry records it."""

    def test_when_sources_and_no_tombstone_then_live(self) -> None:
        assert make_asset_view().is_live

    def test_when_retired_or_without_sources_then_not_live(self) -> None:
        assert not make_asset_view(retired_at=datetime(2026, 1, 1, tzinfo=UTC)).is_live
        assert not make_asset_view(has_sources=False).is_live


class TestGivenSceneView:
    """A scene's span."""

    def test_when_duration_asked_then_end_minus_start(self) -> None:
        scene = make_scene_set().scenes[2]

        assert scene.duration_seconds == 4.5


class TestGivenTimestamps:
    """Millisecond precision of the stores."""

    def test_when_truncated_then_microseconds_dropped(self) -> None:
        moment = datetime(2026, 10, 28, 9, 0, 0, 123_456, tzinfo=UTC)

        assert models.truncate_to_millisecond(moment).microsecond == 123_000


class TestGivenReport:
    """Counting outcomes of a pass."""

    def test_when_outcomes_counted_then_each_field_tallied(self) -> None:
        report = models.EnrichmentReport.counting(
            ["enriched", "enriched", "duplicate", "dropped", "deferred", "failed"]
        )

        assert report == models.EnrichmentReport(
            enriched=2, duplicate=1, dropped=1, deferred=1, failed=1
        )
        assert str(report) == "enriched=2 duplicate=1 dropped=1 deferred=1 failed=1"

    def test_when_no_outcomes_then_zeroes(self) -> None:
        assert models.EnrichmentReport.counting([]) == models.EnrichmentReport()
