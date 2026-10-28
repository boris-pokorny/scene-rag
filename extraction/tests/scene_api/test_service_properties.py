"""Properties of the listing: every live set once, in order, whatever the page size."""

from collections.abc import Iterable, Sequence

from hypothesis import given, settings
from hypothesis import strategies as st

from scene_rag.extraction.scene_api import models
from scene_rag.extraction.scene_api.encoding import encode_scene_record
from scene_rag.extraction.scene_api.service import SceneCatalogService
from tests.support.scene_api_fakes import (
    TRANSCRIPT_FP,
    FakeEnrichmentReader,
    enrich_all,
    make_enrichment_view,
    make_scene_api_harness,
    make_scene_set_view,
    sha256_version,
)

SETS = [make_scene_set_view(sha256_version(bytes([i]))) for i in range(6)]


@given(
    live=st.lists(st.booleans(), min_size=len(SETS), max_size=len(SETS)),
    limit=st.integers(min_value=1, max_value=len(SETS) + 1),
    page_size=st.integers(min_value=1, max_value=len(SETS) + 1),
    page_fill=st.integers(min_value=1, max_value=len(SETS) + 1),
)
@settings(max_examples=80)
def test_when_walked_then_every_live_set_exactly_once_in_order(
    live: list[bool], limit: int, page_size: int, page_fill: int
) -> None:
    harness = make_scene_api_harness(page_size=page_size, page_fill=page_fill)
    for scene_set, is_live in zip(SETS, live, strict=True):
        harness.seed(scene_set, live=is_live)

    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        page = harness.service.list_scene_sets(models.SceneSetQuery(after=cursor, limit=limit))
        pages += 1
        assert len(page.items) <= min(limit, page_size)
        seen += [item.version for item in page.items]
        if page.next is None:
            break
        assert page.next > (cursor or "")
        cursor = page.next

    expected = sorted(s.version for s, is_live in zip(SETS, live, strict=True) if is_live)
    assert seen == expected
    assert pages <= -(-len(SETS) // min(limit, page_size)) + 1


class LooseEnrichmentReader(FakeEnrichmentReader):
    """A reader that ignores the pins: proves the service filters records on its own."""

    def list_for_scenes(
        self,
        scene_set_version: str,
        scene_indices: Iterable[int],
        pins: Sequence[models.EnrichmentPin] = (),
    ) -> Sequence[models.EnrichmentView]:
        return super().list_for_scenes(scene_set_version, scene_indices)


ENRICHERS = ["transcript", "description"]
FINGERPRINTS = [TRANSCRIPT_FP, "ffffffffffff", "000000000000"]


@given(
    written=st.sets(st.tuples(st.sampled_from(ENRICHERS), st.sampled_from(FINGERPRINTS))),
    pinned=st.sets(st.tuples(st.sampled_from(ENRICHERS), st.sampled_from(FINGERPRINTS))),
)
@settings(max_examples=60)
def test_when_pins_pushed_to_the_reader_then_records_unchanged(
    written: set[tuple[str, str]], pinned: set[tuple[str, str]]
) -> None:
    scene_set = SETS[0]
    records = [
        record
        for enricher, fingerprint in sorted(written)
        for record in enrich_all(scene_set, enricher, fingerprint)
    ]
    pins = tuple(models.EnrichmentPin(e, f) for e, f in sorted(pinned))
    strict = make_scene_api_harness()
    strict.seed(scene_set, enrichments=records)
    loose = make_scene_api_harness()
    loose.enrichments = LooseEnrichmentReader()
    loose.service = SceneCatalogService(
        assets=loose.assets,
        scene_sets=loose.scene_sets,
        enrichments=loose.enrichments,
        derivatives=loose.derivatives,
        page_size=loose.page_size,
    )
    loose.seed(scene_set, enrichments=records)

    query = models.SceneQuery(pins=pins)
    assert strict.service.list_scene_records(scene_set.version, query) == (
        loose.service.list_scene_records(scene_set.version, query)
    )
    assert strict.service.get_scene_record(scene_set.version, 1, pins) == (
        loose.service.get_scene_record(scene_set.version, 1, pins)
    )
    assert strict.enrichments.queries[-1][2] == pins


@given(data=st.data())
@settings(max_examples=40)
def test_when_same_inputs_then_same_encoding_and_a_new_enrichment_changes_it(
    data: st.DataObject,
) -> None:
    scene_set = SETS[0]
    indices = data.draw(st.sets(st.integers(0, len(scene_set.scenes) - 1), min_size=1))
    first = make_scene_api_harness()
    second = make_scene_api_harness()
    records = [make_enrichment_view(scene_set, i) for i in sorted(indices)]
    first.seed(scene_set, enrichments=records)
    second.seed(scene_set, enrichments=reversed(records))

    index = data.draw(st.sampled_from(sorted(indices)))
    before = first.service.get_scene_record(scene_set.version, index, [])
    assert encode_scene_record(before) == encode_scene_record(
        second.service.get_scene_record(scene_set.version, index, [])
    )

    first.enrichments.add(*enrich_all(scene_set, "description", "ffffffffffff"))
    after = first.service.get_scene_record(scene_set.version, index, [])
    assert after.version != before.version
    pinned = first.service.get_scene_record(
        scene_set.version, index, [models.EnrichmentPin("transcript", TRANSCRIPT_FP)]
    )
    assert pinned == before
