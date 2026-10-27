from datetime import UTC, datetime, timedelta, timezone

from scene_rag.extraction.ingestion.models import truncate_to_millisecond


class TestGivenTimestamp:
    """Datetimes as the clock and the connectors produce them."""

    def test_when_truncated_then_microseconds_dropped_and_zone_kept(self) -> None:
        moment = datetime(2026, 10, 27, 9, 0, 0, 123_456, tzinfo=UTC)

        assert truncate_to_millisecond(moment) == moment.replace(microsecond=123_000)

    def test_when_already_at_millisecond_then_unchanged(self) -> None:
        offset = timezone(timedelta(hours=2))
        moment = datetime(2026, 10, 27, 9, 0, 0, 5_000, tzinfo=offset)

        assert truncate_to_millisecond(moment) == moment
        assert truncate_to_millisecond(moment).tzinfo is offset
