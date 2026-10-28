"""Value objects of the chunking step."""

from datetime import UTC, datetime

import pytest

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import UpstreamConfig
from tests.support.chunking_fakes import DESCRIPTION_FP, TRANSCRIPT_FP, make_summary


class TestGivenChunkVersion:
    """The record identifiers derived from inputs and fingerprint."""

    def test_when_same_inputs_in_any_order_then_same_version(self) -> None:
        first = models.chunk_version(["sha256:ab", "sha256:cd"], "0123456789ab", 0, "sha256")

        assert first == models.chunk_version(
            ["sha256:cd", "sha256:ab"], "0123456789ab", 0, "sha256"
        )
        assert first.startswith("sha256:") and len(first) == len("sha256:") + 64

    def test_when_any_input_differs_then_version_differs(self) -> None:
        base = models.chunk_version(["sha256:ab"], "0123456789ab", 0, "sha256")

        assert models.chunk_version(["sha256:ac"], "0123456789ab", 0, "sha256") != base
        assert models.chunk_version(["sha256:ab"], "0123456789ac", 0, "sha256") != base
        assert models.chunk_version(["sha256:ab"], "0123456789ab", 1, "sha256") != base

    def test_when_chunk_set_version_then_depends_on_set_and_fingerprint(self) -> None:
        base = models.chunk_set_version("sha256:ab", "0123456789ab", "sha256")

        assert base == models.chunk_set_version("sha256:ab", "0123456789ab", "sha256")
        assert models.chunk_set_version("sha256:ac", "0123456789ab", "sha256") != base
        assert models.chunk_set_version("sha256:ab", "0123456789ac", "sha256") != base


class TestGivenPins:
    """Parsing and rendering of enrichment pins."""

    def test_when_parsed_then_round_trips(self) -> None:
        pin = models.EnrichmentPin.parse("transcript:e41a9c27b0d5")

        assert pin == models.EnrichmentPin("transcript", "e41a9c27b0d5")
        assert str(pin) == "transcript:e41a9c27b0d5"

    @pytest.mark.parametrize("text", ["transcript", "Transcript:e41a9c27b0d5", "transcript:xyz"])
    def test_when_malformed_then_value_error(self, text: str) -> None:
        with pytest.raises(ValueError, match="enrichment pin"):
            models.EnrichmentPin.parse(text)


class TestGivenResolvePins:
    """Which fingerprints a scene set is chunked under."""

    def test_when_every_required_enricher_complete_then_one_pin_each(self) -> None:
        pins = models.resolve_pins(make_summary(), UpstreamConfig())

        assert pins == (
            models.EnrichmentPin("description", DESCRIPTION_FP),
            models.EnrichmentPin("transcript", TRANSCRIPT_FP),
        )

    def test_when_required_enricher_incomplete_then_not_ready(self) -> None:
        summary = make_summary(
            coverage=[("description", DESCRIPTION_FP, True), ("transcript", TRANSCRIPT_FP, False)]
        )

        assert models.resolve_pins(summary, UpstreamConfig()) is None

    def test_when_required_enricher_absent_then_not_ready(self) -> None:
        summary = make_summary(coverage=[("description", DESCRIPTION_FP, True)])

        assert models.resolve_pins(summary, UpstreamConfig()) is None

    def test_when_several_complete_then_greatest_fingerprint(self) -> None:
        summary = make_summary(
            coverage=[
                ("description", "aaaaaaaaaaaa", True),
                ("description", "bbbbbbbbbbbb", True),
                ("transcript", TRANSCRIPT_FP, True),
            ]
        )

        pins = models.resolve_pins(summary, UpstreamConfig())

        assert pins is not None and pins[0] == models.EnrichmentPin("description", "bbbbbbbbbbbb")

    def test_when_pinned_then_pin_wins_over_greater_fingerprint(self) -> None:
        summary = make_summary(
            coverage=[
                ("description", "aaaaaaaaaaaa", True),
                ("description", "bbbbbbbbbbbb", True),
                ("transcript", TRANSCRIPT_FP, True),
            ]
        )
        upstream = UpstreamConfig(enrichment_pins=("description:aaaaaaaaaaaa",))

        pins = models.resolve_pins(summary, upstream)

        assert pins is not None and pins[0] == models.EnrichmentPin("description", "aaaaaaaaaaaa")

    def test_when_pinned_fingerprint_incomplete_then_not_ready(self) -> None:
        summary = make_summary(
            coverage=[("description", "aaaaaaaaaaaa", False), ("transcript", TRANSCRIPT_FP, True)]
        )
        upstream = UpstreamConfig(enrichment_pins=("description:aaaaaaaaaaaa",))

        assert models.resolve_pins(summary, upstream) is None

    def test_when_fewer_enrichers_required_then_only_those(self) -> None:
        summary = make_summary(coverage=[("description", DESCRIPTION_FP, True)])

        pins = models.resolve_pins(summary, UpstreamConfig(required_enrichers=("description",)))

        assert pins == (models.EnrichmentPin("description", DESCRIPTION_FP),)


class TestGivenTimestamps:
    """Millisecond precision of the stores."""

    def test_when_truncated_then_microseconds_dropped(self) -> None:
        moment = datetime(2026, 10, 28, 9, 0, 0, 123_456, tzinfo=UTC)

        assert models.truncate_to_millisecond(moment).microsecond == 123_000
