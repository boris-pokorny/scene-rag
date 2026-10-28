"""The use cases of the scene-api role: joining the stores into scene records.

Transport-free: `SceneCatalogService` takes queries and returns records, summaries and pages;
the delivery adapter turns them into HTTP. Business rules live here and nowhere else: which
scene sets a listing shows (live asset versions, complete under the consumer's pins), which
enrichments a record includes (every one, or the pinned ones) and how pages advance.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from scene_rag.extraction.scene_api import models
from scene_rag.extraction.scene_api.protocols import (
    AssetVersionReader,
    DerivativeReader,
    EnrichmentReader,
    InvalidQueryError,
    NotFoundError,
    SceneSetReader,
)

logger = logging.getLogger(__name__)

# What a record says of an asset version the registry has no record of. Ingestion never
# deletes a registry record, so the case is a consistency gap, reported rather than hidden.
_ABSENT_MEDIA_TYPE = "application/octet-stream"


class SceneCatalogService:
    """`SceneCatalog` over the role's read views of the stores."""

    def __init__(
        self,
        *,
        assets: AssetVersionReader,
        scene_sets: SceneSetReader,
        enrichments: EnrichmentReader,
        derivatives: DerivativeReader,
        page_size: int,
        page_fill: int = 1,
    ) -> None:
        """Compose the use cases; `bootstrap` is the only caller outside tests.

        Args:
            assets: Read view of the Asset Registry.
            scene_sets: Read view of the Scene Store.
            enrichments: Read view of the Enrichment Store.
            derivatives: Reader of the bytes the scene sets reference.
            page_size: Items a page holds when a request names no limit, and the most it may
                ask for.
            page_fill: Most scans of ``limit`` scene sets one listing request makes to fill
                its page, when scanned sets are dropped as retired or incomplete. One means a
                single scan, so a page is short exactly as often as sets were dropped; more
                trades store reads for fuller pages.

        Raises:
            ValueError: When the page size or the page fill is below one.
        """
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        if page_fill < 1:
            raise ValueError("page_fill must be at least 1")
        self._assets = assets
        self._scene_sets = scene_sets
        self._enrichments = enrichments
        self._derivatives = derivatives
        self._page_size = page_size
        self._page_fill = page_fill

    def list_scene_sets(self, query: models.SceneSetQuery) -> models.Page[models.SceneSetSummary]:
        """Return a page of scene sets; see `SceneCatalog.list_scene_sets`.

        The cursor advances over every scene set scanned, not only those returned, so a page
        may come back short or empty while more follow; ``next`` alone says when the listing
        ends. Up to ``page_fill`` scans of ``limit`` sets each go into one page: scanning
        stops once the page is full, the store is exhausted or the scans are spent.
        """
        limit = self._limit(query.limit)
        items: list[models.SceneSetSummary] = []
        cursor = query.after
        for _ in range(self._page_fill):
            wanted = limit - len(items)
            scanned = list(
                self._scene_sets.list_summaries(
                    after=cursor,
                    limit=wanted + 1,
                    asset_version=query.asset_version,
                    segmentation_fingerprint=query.segmentation_fingerprint,
                )
            )
            more = len(scanned) > wanted
            scanned = scanned[:wanted]
            if not scanned:
                return models.Page(tuple(items), None)
            items.extend(self._listed(scanned, query.pins))
            cursor = scanned[-1].version
            if not more:
                return models.Page(tuple(items), None)
            if len(items) >= limit:
                break
        return models.Page(tuple(items), cursor)

    def _listed(
        self, scanned: Sequence[models.SceneSetSummaryView], pins: Sequence[models.EnrichmentPin]
    ) -> list[models.SceneSetSummary]:
        """Return the summaries of the scanned sets a listing shows: live, complete under pins."""
        assets = self._assets.get_many(summary.asset_version for summary in scanned)
        coverage = self._coverage_by_set(summary.version for summary in scanned)
        items: list[models.SceneSetSummary] = []
        for summary in scanned:
            asset = assets.get(summary.asset_version)
            if asset is None or not asset.is_live:
                continue
            covered = self._coverage_of(summary, coverage.get(summary.version, ()))
            if not self._complete(covered, pins):
                continue
            items.append(self._summary(summary, live=True, covered=covered))
        return items

    def get_scene_set(self, version: str) -> models.SceneSetSummary:
        """Return one scene set's summary; see `SceneCatalog.get_scene_set`."""
        scene_set = self._scene_set(version)
        summary = scene_set.summary
        covered = self._coverage_of(summary, self._coverage_by_set([version]).get(version, ()))
        return self._summary(summary, live=self._asset_of(scene_set).is_live, covered=covered)

    def list_scene_records(
        self, version: str, query: models.SceneQuery
    ) -> models.Page[models.SceneRecord]:
        """Return a page of scene records; see `SceneCatalog.list_scene_records`."""
        limit = self._limit(query.limit)
        scene_set = self._scene_set(version)
        scenes = [
            scene for scene in scene_set.scenes if query.after is None or scene.index > query.after
        ]
        more = len(scenes) > limit
        scenes = scenes[:limit]
        if not scenes:
            return models.Page((), None)
        records = self._records(scene_set, scenes, query.pins)
        return models.Page(records, str(scenes[-1].index) if more else None)

    def get_scene_record(
        self, version: str, index: int, pins: Sequence[models.EnrichmentPin]
    ) -> models.SceneRecord:
        """Return one scene record; see `SceneCatalog.get_scene_record`."""
        scene_set = self._scene_set(version)
        if not 0 <= index < len(scene_set.scenes):
            raise NotFoundError("scene", f"{version}/{index}")
        return self._records(scene_set, [scene_set.scenes[index]], tuple(pins))[0]

    def get_asset_version(self, version: str) -> models.AssetVersionView:
        """Return one asset version; see `SceneCatalog.get_asset_version`."""
        asset = self._assets.get(version)
        if asset is None:
            raise NotFoundError("asset-version", version)
        return asset

    def find_derivative(self, digest: str) -> models.DerivativeRefView:
        """Return a derivative's reference; see `SceneCatalog.find_derivative`."""
        reference = self._scene_sets.find_derivative(digest)
        if reference is None:
            raise NotFoundError("derivative", digest)
        return reference

    def get_derivative(self, digest: str) -> models.DerivativeContent:
        """Return a derivative's verified bytes; see `SceneCatalog.get_derivative`."""
        reference = self.find_derivative(digest)
        return models.DerivativeContent(reference, self._derivatives.read(reference))

    def _limit(self, limit: int | None) -> int:
        if limit is None:
            return self._page_size
        if limit < 1:
            raise InvalidQueryError(f"limit must be at least 1, not {limit}")
        return min(limit, self._page_size)

    def _scene_set(self, version: str) -> models.SceneSetView:
        scene_set = self._scene_sets.get(version)
        if scene_set is None:
            raise NotFoundError("scene-set", version)
        return scene_set

    def _asset_of(self, scene_set: models.SceneSetView) -> models.AssetVersionView:
        asset = self._assets.get(scene_set.asset_version)
        if asset is not None:
            return asset
        logger.warning(
            "scene set %s names asset version %s, which the registry does not hold",
            scene_set.version,
            scene_set.asset_version,
        )
        return models.AssetVersionView(
            version=scene_set.asset_version,
            media_type=_ABSENT_MEDIA_TYPE,
            size_bytes=0,
            sources=(),
            entitlements=frozenset(),
        )

    def _coverage_by_set(self, versions: Iterable[str]) -> Mapping[str, Sequence[models.Coverage]]:
        by_set: defaultdict[str, list[models.Coverage]] = defaultdict(list)
        for coverage in self._enrichments.list_coverage(versions):
            by_set[coverage.scene_set_version].append(coverage)
        return by_set

    @staticmethod
    def _coverage_of(
        summary: models.SceneSetSummaryView, coverage: Iterable[models.Coverage]
    ) -> tuple[models.EnrichmentCoverage, ...]:
        return tuple(
            models.EnrichmentCoverage(
                enricher=item.enricher,
                fingerprint=item.fingerprint,
                scene_count=item.scene_count,
                complete=item.scene_count >= summary.scene_count,
            )
            for item in sorted(coverage, key=lambda item: (item.enricher, item.fingerprint))
        )

    @staticmethod
    def _complete(
        covered: Iterable[models.EnrichmentCoverage], pins: Iterable[models.EnrichmentPin]
    ) -> bool:
        complete = {
            models.EnrichmentPin(item.enricher, item.fingerprint)
            for item in covered
            if item.complete
        }
        return all(pin in complete for pin in pins)

    @staticmethod
    def _summary(
        summary: models.SceneSetSummaryView,
        *,
        live: bool,
        covered: tuple[models.EnrichmentCoverage, ...],
    ) -> models.SceneSetSummary:
        return models.SceneSetSummary(
            version=summary.version,
            asset_version=summary.asset_version,
            segmentation_fingerprint=summary.segmentation_fingerprint,
            scene_count=summary.scene_count,
            created_at=summary.created_at,
            live=live,
            enrichments=covered,
        )

    def _records(
        self,
        scene_set: models.SceneSetView,
        scenes: Sequence[models.SceneView],
        pins: tuple[models.EnrichmentPin, ...],
    ) -> tuple[models.SceneRecord, ...]:
        asset = self._asset_of(scene_set)
        by_scene: defaultdict[int, list[models.EnrichmentView]] = defaultdict(list)
        # The reader narrows to the pins; the check here keeps a looser reader from leaking.
        for record in self._enrichments.list_for_scenes(
            scene_set.version, [scene.index for scene in scenes], pins
        ):
            if not pins or record.pin in pins:
                by_scene[record.scene_index].append(record)
        return tuple(
            self._record(scene_set, asset, scene, by_scene.get(scene.index, [])) for scene in scenes
        )

    @staticmethod
    def _record(
        scene_set: models.SceneSetView,
        asset: models.AssetVersionView,
        scene: models.SceneView,
        enrichments: Iterable[models.EnrichmentView],
    ) -> models.SceneRecord:
        included = tuple(sorted(enrichments, key=lambda record: record.pin))
        return models.SceneRecord(
            version=models.scene_record_version(
                scene_set.version, scene.index, (record.version for record in included)
            ),
            scene_set_version=scene_set.version,
            asset_version=scene_set.asset_version,
            segmentation_fingerprint=scene_set.segmentation_fingerprint,
            live=asset.is_live,
            asset=asset,
            media=scene_set.media,
            scene=scene,
            enrichments=included,
        )
