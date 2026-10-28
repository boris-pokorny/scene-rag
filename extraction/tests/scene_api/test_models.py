"""Record versions and request pins of the scene-api role."""

import pytest

from scene_rag.extraction.scene_api import models
from tests.support.scene_api_fakes import make_asset_view, retired, sha256_version

SET = sha256_version(b"set")
A = sha256_version(b"a")
B = sha256_version(b"b")


class TestGivenSceneRecordVersion:
    """The record version derives from the scene set, the index and the enrichment versions."""

    def test_when_enrichments_reordered_then_same_version(self) -> None:
        assert models.scene_record_version(SET, 0, [A, B]) == models.scene_record_version(
            SET, 0, [B, A]
        )

    def test_when_enrichment_added_then_version_changes(self) -> None:
        assert models.scene_record_version(SET, 0, [A]) != models.scene_record_version(
            SET, 0, [A, B]
        )

    def test_when_index_differs_then_version_changes(self) -> None:
        assert models.scene_record_version(SET, 0, [A]) != models.scene_record_version(SET, 1, [A])

    def test_when_no_enrichments_then_still_versioned(self) -> None:
        version = models.scene_record_version(SET, 2, [])

        assert version.startswith("sha256:") and len(version) == len("sha256:") + 64

    def test_when_other_algorithm_then_prefixed_with_it(self) -> None:
        assert models.scene_record_version(SET, 0, [A], "sha1").startswith("sha1:")


class TestGivenEnrichmentPin:
    """Parsing ``<enricher>:<fingerprint>``."""

    def test_when_well_formed_then_parsed(self) -> None:
        pin = models.EnrichmentPin.parse("transcript:0123456789ab")

        assert pin == models.EnrichmentPin("transcript", "0123456789ab")
        assert str(pin) == "transcript:0123456789ab"

    @pytest.mark.parametrize(
        "text",
        [
            "transcript",
            "transcript:",
            ":0123456789ab",
            "Transcript:0123456789ab",
            "transcript:0123456789",
            "transcript:0123456789abcd",
            "transcript:0123456789AB",
            "transcript:0123456789ab:extra",
        ],
    )
    def test_when_malformed_then_value_error(self, text: str) -> None:
        with pytest.raises(ValueError, match="enrichment pin"):
            models.EnrichmentPin.parse(text)

    def test_when_pins_compared_then_ordered_by_enricher_then_fingerprint(self) -> None:
        pins = [
            models.EnrichmentPin("transcript", "000000000000"),
            models.EnrichmentPin("description", "ffffffffffff"),
            models.EnrichmentPin("description", "000000000000"),
        ]

        assert sorted(pins) == [pins[2], pins[1], pins[0]]


class TestGivenAssetVersionView:
    """Liveness of an asset version."""

    def test_when_held_and_not_retired_then_live(self) -> None:
        assert make_asset_view().is_live

    def test_when_retired_then_not_live(self) -> None:
        assert not retired(make_asset_view()).is_live

    def test_when_no_sources_then_not_live(self) -> None:
        view = models.AssetVersionView("sha256:ab", "video/mp4", 1, (), frozenset())

        assert not view.is_live


class TestGivenSceneView:
    """A scene's derivative references."""

    def test_when_listed_then_keyframes_before_clip(self) -> None:
        from tests.support.scene_api_fakes import make_scenes

        scene = make_scenes()[0]

        assert scene.derivatives == (scene.keyframes[0].derivative, scene.audio_clip)

    def test_when_no_clip_then_keyframes_only(self) -> None:
        from tests.support.scene_api_fakes import make_scenes

        scene = make_scenes()[1]

        assert scene.audio_clip is None
        assert scene.derivatives == (scene.keyframes[0].derivative,)
