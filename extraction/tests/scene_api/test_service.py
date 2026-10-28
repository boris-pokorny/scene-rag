"""The use cases of the scene-api role over fakes."""

import dataclasses
from collections.abc import Sequence

import pytest

from scene_rag.extraction.scene_api import models
from scene_rag.extraction.scene_api.protocols import (
    DerivativeUnavailableError,
    InvalidQueryError,
    NotFoundError,
)
from scene_rag.extraction.scene_api.service import SceneCatalogService
from tests.support.scene_api_fakes import (
    DESCRIPTION_FP,
    SEGMENTATION_FP,
    TRANSCRIPT_FP,
    FakeSceneSetReader,
    SceneApiHarness,
    enrich_all,
    make_enrichment_view,
    make_scene_api_harness,
    make_scene_set_view,
    sha256_version,
)

FIRST = make_scene_set_view(sha256_version(b"first"))
SECOND = make_scene_set_view(sha256_version(b"second"))
THIRD = make_scene_set_view(sha256_version(b"third"))
RECUT = make_scene_set_view(sha256_version(b"first"), "bbbbbbbbbbbb")
ORDERED = sorted((FIRST, SECOND, THIRD, RECUT), key=lambda s: s.version)
TRANSCRIPT = models.EnrichmentPin("transcript", TRANSCRIPT_FP)
DESCRIPTION = models.EnrichmentPin("description", DESCRIPTION_FP)


def versions(page: models.Page[models.SceneSetSummary]) -> list[str]:
    return [item.version for item in page.items]


class TestGivenLiveAndRetiredSets:
    """Four scene sets, one of a retired asset version, two of one asset under two cuts."""

    @pytest.fixture
    def scene_api_harness(self) -> SceneApiHarness:
        harness = make_scene_api_harness(page_size=10)
        harness.seed(FIRST, SECOND, RECUT)
        harness.seed(THIRD, live=False)
        return harness

    def test_when_listed_then_live_sets_in_version_order(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        page = scene_api_harness.service.list_scene_sets(models.SceneSetQuery())

        assert versions(page) == [s.version for s in ORDERED if s is not THIRD]
        assert page.next is None
        assert all(item.live for item in page.items)

    def test_when_paged_then_cursor_is_last_scanned_version(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        service = scene_api_harness.service
        seen: list[str] = []
        cursor: str | None = None
        pages = 0
        while True:
            page = service.list_scene_sets(models.SceneSetQuery(after=cursor, limit=1))
            pages += 1
            seen += versions(page)
            assert len(page.items) <= 1
            if page.next is None:
                break
            cursor = page.next

        assert seen == [s.version for s in ORDERED if s is not THIRD]
        assert pages == 4  # the retired set's page comes back empty, with a cursor

    def test_when_retired_set_paged_alone_then_empty_page_with_cursor(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        before = [s.version for s in ORDERED if s.version < THIRD.version]

        page = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(after=before[-1] if before else None, limit=1)
        )

        assert page.items == ()
        assert page.next == THIRD.version

    def test_when_filtered_by_asset_version_then_both_cuts(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        page = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(asset_version=FIRST.asset_version)
        )

        assert set(versions(page)) == {FIRST.version, RECUT.version}

    def test_when_filtered_by_segmentation_then_one_cut(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        page = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(segmentation_fingerprint="bbbbbbbbbbbb")
        )

        assert versions(page) == [RECUT.version]

    def test_when_limit_exceeds_page_size_then_clamped(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        harness = make_scene_api_harness(page_size=2)
        harness.seed(FIRST, SECOND, RECUT)

        page = harness.service.list_scene_sets(models.SceneSetQuery(limit=50))

        assert len(page.items) == 2
        assert page.next == sorted(s.version for s in (FIRST, SECOND, RECUT))[1]

    @pytest.mark.parametrize("limit", [0, -1])
    def test_when_limit_below_one_then_invalid(
        self, scene_api_harness: SceneApiHarness, limit: int
    ) -> None:
        with pytest.raises(InvalidQueryError, match="limit"):
            scene_api_harness.service.list_scene_sets(models.SceneSetQuery(limit=limit))

    def test_when_retired_set_fetched_then_served_as_not_live(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        summary = scene_api_harness.service.get_scene_set(THIRD.version)

        assert summary.version == THIRD.version
        assert summary.live is False
        assert summary.scene_count == 3
        assert summary.enrichments == ()

    def test_when_unknown_set_fetched_then_not_found(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        with pytest.raises(NotFoundError) as exc_info:
            scene_api_harness.service.get_scene_set("sha256:0000")

        assert (exc_info.value.kind, exc_info.value.key) == ("scene-set", "sha256:0000")

    def test_when_nothing_matches_then_empty_last_page(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        page = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(asset_version="sha256:0000")
        )

        assert page == models.Page((), None)


class TestGivenPins:
    """Two sets under mixed coverage.

    FIRST is fully transcribed and described, and a newer description fingerprint covers one
    scene of it; SECOND is fully transcribed but only one scene of it is described.
    """

    NEW_DESCRIPTION = models.EnrichmentPin("description", "ffffffffffff")

    @pytest.fixture
    def scene_api_harness(self) -> SceneApiHarness:
        harness = make_scene_api_harness(page_size=10)
        harness.seed(
            FIRST,
            enrichments=(
                *enrich_all(FIRST, "transcript", TRANSCRIPT_FP),
                *enrich_all(FIRST, "description", DESCRIPTION_FP),
                make_enrichment_view(FIRST, 0, "description", self.NEW_DESCRIPTION.fingerprint),
            ),
        )
        harness.seed(
            SECOND,
            enrichments=(
                *enrich_all(SECOND, "transcript", TRANSCRIPT_FP),
                make_enrichment_view(SECOND, 1, "description", DESCRIPTION_FP),
            ),
        )
        return harness

    def test_when_unpinned_then_every_set_with_full_coverage(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        page = scene_api_harness.service.list_scene_sets(models.SceneSetQuery())

        by_version = {item.version: item for item in page.items}
        assert set(by_version) == {FIRST.version, SECOND.version}
        assert by_version[FIRST.version].enrichments == (
            models.EnrichmentCoverage("description", DESCRIPTION_FP, 3, True),
            models.EnrichmentCoverage("description", "ffffffffffff", 1, False),
            models.EnrichmentCoverage("transcript", TRANSCRIPT_FP, 3, True),
        )
        assert by_version[SECOND.version].enrichments == (
            models.EnrichmentCoverage("description", DESCRIPTION_FP, 1, False),
            models.EnrichmentCoverage("transcript", TRANSCRIPT_FP, 3, True),
        )

    def test_when_pinned_to_complete_enrichers_then_only_complete_sets(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        both = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(pins=(TRANSCRIPT, DESCRIPTION))
        )
        transcript_only = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(pins=(TRANSCRIPT,))
        )
        new_description = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(pins=(self.NEW_DESCRIPTION,))
        )
        unknown = scene_api_harness.service.list_scene_sets(
            models.SceneSetQuery(pins=(models.EnrichmentPin("ocr", "000000000000"),))
        )

        assert versions(both) == [FIRST.version]
        assert set(versions(transcript_only)) == {FIRST.version, SECOND.version}
        assert versions(new_description) == []
        assert versions(unknown) == []

    def test_when_records_pinned_then_only_pinned_enrichments(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        record = scene_api_harness.service.get_scene_record(FIRST.version, 0, [DESCRIPTION])

        assert [item.pin for item in record.enrichments] == [DESCRIPTION]

    def test_when_records_pinned_then_pins_reach_the_reader(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        scene_api_harness.service.list_scene_records(
            FIRST.version, models.SceneQuery(pins=(TRANSCRIPT, DESCRIPTION))
        )
        scene_api_harness.service.get_scene_record(FIRST.version, 1, [TRANSCRIPT])

        assert scene_api_harness.enrichments.queries == [
            (FIRST.version, frozenset({0, 1, 2}), (TRANSCRIPT, DESCRIPTION)),
            (FIRST.version, frozenset({1}), (TRANSCRIPT,)),
        ]

    def test_when_records_unpinned_then_every_enrichment_sorted(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        record = scene_api_harness.service.get_scene_record(FIRST.version, 0, [])

        assert [item.pin for item in record.enrichments] == [
            DESCRIPTION,
            self.NEW_DESCRIPTION,
            TRANSCRIPT,
        ]

    def test_when_new_fingerprint_backfilled_then_pinned_version_unchanged(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        service = scene_api_harness.service
        before = service.get_scene_record(FIRST.version, 1, [TRANSCRIPT, DESCRIPTION])
        unpinned_before = service.get_scene_record(FIRST.version, 1, [])

        scene_api_harness.enrichments.add(
            make_enrichment_view(FIRST, 1, "description", self.NEW_DESCRIPTION.fingerprint)
        )
        after = service.get_scene_record(FIRST.version, 1, [TRANSCRIPT, DESCRIPTION])
        unpinned_after = service.get_scene_record(FIRST.version, 1, [])

        assert after == before
        assert unpinned_after.version != unpinned_before.version


class TestGivenSceneSet:
    """One live scene set of three scenes, transcribed."""

    @pytest.fixture
    def scene_api_harness(self) -> SceneApiHarness:
        harness = make_scene_api_harness(page_size=2)
        harness.seed(FIRST, enrichments=enrich_all(FIRST, "transcript", TRANSCRIPT_FP))
        return harness

    def test_when_records_listed_then_paged_by_scene_index(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        service = scene_api_harness.service

        first = service.list_scene_records(FIRST.version, models.SceneQuery())
        second = service.list_scene_records(FIRST.version, models.SceneQuery(after=1))

        assert [r.scene.index for r in first.items] == [0, 1]
        assert first.next == "1"
        assert [r.scene.index for r in second.items] == [2]
        assert second.next is None

    def test_when_record_built_then_everything_denormalised(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        record = scene_api_harness.service.get_scene_record(FIRST.version, 2, [])
        transcript = make_enrichment_view(FIRST, 2)

        assert record == models.SceneRecord(
            version=models.scene_record_version(FIRST.version, 2, [transcript.version]),
            scene_set_version=FIRST.version,
            asset_version=FIRST.asset_version,
            segmentation_fingerprint=SEGMENTATION_FP,
            live=True,
            asset=scene_api_harness.assets.records[FIRST.asset_version],
            media=FIRST.media,
            scene=FIRST.scenes[2],
            enrichments=(transcript,),
        )

    def test_when_after_beyond_last_scene_then_empty_last_page(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        page = scene_api_harness.service.list_scene_records(
            FIRST.version, models.SceneQuery(after=7)
        )

        assert page == models.Page((), None)

    def test_when_index_out_of_range_then_not_found(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        for index in (3, -1):
            with pytest.raises(NotFoundError) as exc_info:
                scene_api_harness.service.get_scene_record(FIRST.version, index, [])
            assert exc_info.value.kind == "scene"

    def test_when_unknown_set_then_not_found(self, scene_api_harness: SceneApiHarness) -> None:
        with pytest.raises(NotFoundError, match="scene-set"):
            scene_api_harness.service.list_scene_records("sha256:0000", models.SceneQuery())
        with pytest.raises(NotFoundError, match="scene-set"):
            scene_api_harness.service.get_scene_record("sha256:0000", 0, [])

    def test_when_limit_below_one_then_invalid(self, scene_api_harness: SceneApiHarness) -> None:
        with pytest.raises(InvalidQueryError):
            scene_api_harness.service.list_scene_records(FIRST.version, models.SceneQuery(limit=0))

    def test_when_scene_without_enrichments_then_empty_tuple(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        scene_api_harness.seed(SECOND)

        record = scene_api_harness.service.get_scene_record(SECOND.version, 0, [])

        assert record.enrichments == ()
        assert record.version == models.scene_record_version(SECOND.version, 0, [])

    def test_when_asset_retired_then_records_not_live(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        scene_api_harness.assets.retire(FIRST.asset_version)

        record = scene_api_harness.service.get_scene_record(FIRST.version, 0, [])

        assert record.live is False
        assert record.asset.entitlements == frozenset()
        assert record.asset.sources == ()

    def test_when_registry_lacks_asset_then_served_not_live_and_logged(
        self, scene_api_harness: SceneApiHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        del scene_api_harness.assets.records[FIRST.asset_version]

        with caplog.at_level("WARNING"):
            record = scene_api_harness.service.get_scene_record(FIRST.version, 0, [])
            summary = scene_api_harness.service.get_scene_set(FIRST.version)

        assert record.live is False and summary.live is False
        assert record.asset == models.AssetVersionView(
            FIRST.asset_version, "application/octet-stream", 0, (), frozenset()
        )
        assert FIRST.asset_version in caplog.text


class TestGivenAssetVersions:
    """Asset versions are served whether live or retired."""

    def test_when_known_then_returned(self, scene_api_harness: SceneApiHarness) -> None:
        scene_api_harness.seed(THIRD, live=False)

        asset = scene_api_harness.service.get_asset_version(THIRD.asset_version)

        assert asset.version == THIRD.asset_version
        assert not asset.is_live

    def test_when_unknown_then_not_found(self, scene_api_harness: SceneApiHarness) -> None:
        with pytest.raises(NotFoundError, match="asset-version"):
            scene_api_harness.service.get_asset_version("sha256:0000")


class TestGivenDerivatives:
    """Bytes are served for digests some scene set references."""

    def test_when_referenced_then_bytes_with_reference(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        scene_api_harness.seed(FIRST)
        clip = FIRST.scenes[0].audio_clip
        assert clip is not None

        content = scene_api_harness.service.get_derivative(clip.digest)

        assert content == models.DerivativeContent(clip, b"wav")

    def test_when_reference_found_then_no_bytes_read(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        scene_api_harness.seed(FIRST)
        clip = FIRST.scenes[0].audio_clip

        reference = scene_api_harness.service.find_derivative(
            FIRST.scenes[0].keyframes[0].derivative.digest
        )

        assert reference == FIRST.scenes[0].keyframes[0].derivative
        assert reference != clip
        assert scene_api_harness.derivatives.reads == []

    def test_when_unreferenced_then_not_found(self, scene_api_harness: SceneApiHarness) -> None:
        scene_api_harness.seed(FIRST)

        for lookup in (
            scene_api_harness.service.get_derivative,
            scene_api_harness.service.find_derivative,
        ):
            with pytest.raises(NotFoundError, match="derivative"):
                lookup("sha256:0000")
        assert scene_api_harness.derivatives.reads == []

    def test_when_file_unavailable_then_error_propagates(
        self, scene_api_harness: SceneApiHarness
    ) -> None:
        scene_api_harness.seed(FIRST)
        keyframe = FIRST.scenes[0].keyframes[0].derivative
        del scene_api_harness.derivatives.blobs[keyframe.digest]

        with pytest.raises(DerivativeUnavailableError):
            scene_api_harness.service.get_derivative(keyframe.digest)


class CountingSceneSetReader(FakeSceneSetReader):
    """The fake, counting the scans a listing makes."""

    scans = 0

    def list_summaries(
        self,
        *,
        after: str | None,
        limit: int,
        asset_version: str | None,
        segmentation_fingerprint: str | None,
    ) -> Sequence[models.SceneSetSummaryView]:
        self.scans += 1
        return super().list_summaries(
            after=after,
            limit=limit,
            asset_version=asset_version,
            segmentation_fingerprint=segmentation_fingerprint,
        )


class TestGivenPageFill:
    """Six sets, the middle four retired: a page fills across scans up to the fill."""

    SETS = sorted(
        (make_scene_set_view(sha256_version(bytes([i]))) for i in range(6)), key=lambda s: s.version
    )

    def seeded(self, page_fill: int) -> SceneApiHarness:
        harness = make_scene_api_harness(page_size=10, page_fill=page_fill)
        harness.seed(self.SETS[0], self.SETS[5])
        harness.seed(*self.SETS[1:5], live=False)
        return harness

    def test_when_fill_is_one_then_page_short_after_one_scan(self) -> None:
        page = self.seeded(page_fill=1).service.list_scene_sets(models.SceneSetQuery(limit=2))

        assert versions(page) == [self.SETS[0].version]
        assert page.next == self.SETS[1].version

    def test_when_fill_allows_then_later_scans_complete_the_page(self) -> None:
        page = self.seeded(page_fill=10).service.list_scene_sets(models.SceneSetQuery(limit=2))

        assert versions(page) == [self.SETS[0].version, self.SETS[5].version]
        assert page.next is None

    def test_when_scans_spent_then_cursor_is_last_scanned(self) -> None:
        page = self.seeded(page_fill=2).service.list_scene_sets(models.SceneSetQuery(limit=2))

        assert versions(page) == [self.SETS[0].version]
        assert page.next == self.SETS[2].version  # scanned 0,1 then 2 (one short of the page)

    def test_when_page_full_then_no_further_scan(self) -> None:
        harness = make_scene_api_harness(page_size=10, page_fill=10)
        harness.seed(*self.SETS)
        scene_sets = CountingSceneSetReader(*harness.scene_sets.scene_sets.values())
        service = SceneCatalogService(
            assets=harness.assets,
            scene_sets=scene_sets,
            enrichments=harness.enrichments,
            derivatives=harness.derivatives,
            page_size=10,
            page_fill=10,
        )

        page = service.list_scene_sets(models.SceneSetQuery(limit=2))

        assert versions(page) == [s.version for s in self.SETS[:2]]
        assert page.next == self.SETS[1].version
        assert scene_sets.scans == 1


class TestGivenPageSize:
    """The page size and page fill are validated at construction."""

    @pytest.mark.parametrize(
        "page_size, page_fill, field", [(0, 1, "page_size"), (1, 0, "page_fill")]
    )
    def test_when_below_one_then_value_error(
        self, scene_api_harness: SceneApiHarness, page_size: int, page_fill: int, field: str
    ) -> None:
        with pytest.raises(ValueError, match=field):
            SceneCatalogService(
                assets=scene_api_harness.assets,
                scene_sets=scene_api_harness.scene_sets,
                enrichments=scene_api_harness.enrichments,
                derivatives=scene_api_harness.derivatives,
                page_size=page_size,
                page_fill=page_fill,
            )

    def test_when_scenes_tuple_replaced_then_summary_follows(self) -> None:
        shorter = dataclasses.replace(FIRST, scenes=FIRST.scenes[:1])

        assert shorter.summary.scene_count == 1
