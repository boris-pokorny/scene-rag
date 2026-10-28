"""The published JSON shapes: field names, sorting, dates and derivative paths."""

from datetime import UTC, datetime, timedelta, timezone

from scene_rag.extraction.scene_api import encoding, models
from tests.support.scene_api_fakes import (
    T0,
    make_asset_view,
    make_enrichment_view,
    make_record,
    make_scene_set_view,
    make_source,
    retired,
    sha256_version,
)


class TestGivenRecord:
    """A scene record with one keyframe, one clip and one enrichment."""

    def test_when_encoded_then_derivatives_carry_api_paths_not_uris(self) -> None:
        record = make_record()

        encoded = encoding.encode_scene_record(record)

        keyframe = encoded["scene"]["keyframes"][0]
        clip = encoded["scene"]["audio_clip"]
        digest = record.scene.keyframes[0].derivative.digest
        assert keyframe == {
            "digest": digest,
            "media_type": "image/jpeg",
            "size_bytes": 4,
            "url": f"/derivatives/{digest}",
            "frame": 12,
            "seconds": 0.48,
        }
        assert clip["url"] == f"/derivatives/{clip['digest']}"
        assert "uri" not in str(encoded)

    def test_when_encoded_then_asset_has_ids_and_sorted_entitlements(self) -> None:
        asset = make_asset_view(
            sha256_version(b"abcd"),
            make_source("asset-0000000000000002", "library", "talk.mp4"),
            make_source(),
            entitlements=frozenset({"team-platform", "public"}),
        )
        record = make_record(asset=asset)

        encoded = encoding.encode_scene_record(record)

        assert encoded["asset"] == {
            "media_type": "video/mp4",
            "size_bytes": 4,
            "entitlements": ["public", "team-platform"],
            "sources": [
                {
                    "source_id": "library",
                    "asset_id": "asset-0000000000000002",
                    "filename": "talk.mp4",
                },
                {
                    "source_id": "local",
                    "asset_id": "asset-0000000000000001",
                    "filename": "clip.mp4",
                },
            ],
        }
        assert encoded["live"] is True
        assert encoded["schema_version"] == 1

    def test_when_scene_without_clip_then_null(self) -> None:
        scene_set = make_scene_set_view()
        record = make_record(scene_set, 1, enrichments=())

        encoded = encoding.encode_scene_record(record)

        assert encoded["scene"]["audio_clip"] is None
        assert encoded["enrichments"] == []

    def test_when_enrichment_encoded_then_model_and_segments(self) -> None:
        scene_set = make_scene_set_view()
        item = make_enrichment_view(scene_set, 0)

        encoded = encoding.encode_enrichment(item)

        assert encoded == {
            "version": item.version,
            "enricher": "transcript",
            "fingerprint": item.fingerprint,
            "classification": "internal",
            "model": {"name": "transcript-model", "digest": "sha256:" + "ab" * 32},
            "segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 4.0,
                    "text": "transcript of 0",
                    "language": "en",
                    "confidence": 0.9,
                }
            ],
        }


class TestGivenSummaryAndAsset:
    """Summaries, asset versions, pages and errors."""

    def test_when_summary_encoded_then_coverage_listed(self) -> None:
        summary = models.SceneSetSummary(
            version="sha256:ab",
            asset_version="sha256:cd",
            segmentation_fingerprint="0123456789ab",
            scene_count=3,
            created_at=T0,
            live=True,
            enrichments=(models.EnrichmentCoverage("transcript", "e41a9c27b0d5", 2, False),),
        )

        assert encoding.encode_scene_set_summary(summary) == {
            "version": "sha256:ab",
            "asset_version": "sha256:cd",
            "segmentation_fingerprint": "0123456789ab",
            "scene_count": 3,
            "created_at": "2026-10-27T09:00:00Z",
            "live": True,
            "enrichments": [
                {
                    "enricher": "transcript",
                    "fingerprint": "e41a9c27b0d5",
                    "scene_count": 2,
                    "complete": False,
                }
            ],
        }

    def test_when_retired_asset_encoded_then_tombstone_rendered(self) -> None:
        asset = retired(make_asset_view(), datetime(2026, 10, 28, 12, 30, 15, 250000, tzinfo=UTC))

        encoded = encoding.encode_asset_version(asset)

        assert encoded["live"] is False
        assert encoded["retired_at"] == "2026-10-28T12:30:15.250000Z"
        assert encoded["sources"] == [] and encoded["entitlements"] == []

    def test_when_moment_in_another_zone_then_rendered_in_utc(self) -> None:
        moment = datetime(2026, 10, 28, 14, 30, tzinfo=timezone(timedelta(hours=2)))

        assert encoding.encode_datetime(moment) == "2026-10-28T12:30:00Z"

    def test_when_live_asset_encoded_then_null_tombstone(self) -> None:
        encoded = encoding.encode_asset_version(make_asset_view())

        assert encoded["retired_at"] is None and encoded["live"] is True

    def test_when_page_encoded_then_items_and_cursor(self) -> None:
        page = models.Page((1, 2), "next")

        assert encoding.encode_page(page, lambda n: {"n": n}) == {
            "items": [{"n": 1}, {"n": 2}],
            "next": "next",
        }

    def test_when_error_encoded_then_code_and_message(self) -> None:
        assert encoding.encode_error("not_found", "no scene set x") == {
            "error": {"code": "not_found", "message": "no scene set x"}
        }
