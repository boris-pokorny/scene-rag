"""Invariants of the pass over arbitrary registries and partially enriched stores."""

from hypothesis import given, settings
from hypothesis import strategies as st

from scene_rag.extraction.enrichment.dispatch import ThreadDispatcher
from scene_rag.extraction.enrichment.models import EnrichmentReport
from tests.support.enrichment_fakes import (
    make_enrichment,
    make_enrichment_harness,
    make_scene_set,
    sha256_version,
)

VERSIONS = [sha256_version(bytes([i])) for i in range(4)]
SCENE_SETS = {
    version: make_scene_set(version, bounds=((0.0, 1.0), (1.0, 2.0)) if i % 2 else ((0.0, 1.5),))
    for i, version in enumerate(VERSIONS)
}
ENRICHERS = ("description", "transcript")

registries = st.dictionaries(st.sampled_from(VERSIONS), st.booleans())
done_before = st.sets(
    st.tuples(st.sampled_from(VERSIONS), st.sampled_from(ENRICHERS), st.integers(0, 1))
)


def expected_keys(live: set[str]) -> set[tuple[str, str, int]]:
    return {
        (SCENE_SETS[v].version, name, scene.index)
        for v in live
        for name in ENRICHERS
        for scene in SCENE_SETS[v].scenes
    }


class TestGivenAnyRegistry:
    """Live and retired versions in any combination, some scenes already enriched."""

    @given(registry=registries, done=done_before)
    @settings(max_examples=40)
    def test_when_two_passes_run_then_exactly_one_record_per_live_scene_and_enricher(
        self, registry: dict[str, bool], done: set[tuple[str, str, int]]
    ) -> None:
        harness = make_enrichment_harness()
        for version, is_live in registry.items():
            harness.seed(SCENE_SETS[version], live=is_live)
        fingerprints = harness.service.fingerprints
        pre_existing = 0
        for version, name, index in done:
            if index < len(SCENE_SETS[version].scenes):
                harness.store.put(
                    make_enrichment(SCENE_SETS[version], index, name, fingerprints[name])
                )
                pre_existing += 1
        live_versions = {v for v, is_live in registry.items() if is_live}

        first = harness.scheduler.run_once()
        second = harness.scheduler.run_once()

        records = harness.store.snapshot().values()
        live_records = {
            (r.scene_set_version, r.enricher, r.scene_index)
            for r in records
            if r.asset_version in live_versions
        }
        assert live_records == expected_keys(live_versions)
        assert len(records) == pre_existing + len(harness.writer.commands)
        assert second == EnrichmentReport()
        assert first.failed == first.dropped == first.deferred == 0

    @given(registry=registries)
    @settings(max_examples=25, deadline=None)
    def test_when_threaded_then_same_records_as_serial(self, registry: dict[str, bool]) -> None:
        serial = make_enrichment_harness()
        for version, live in registry.items():
            serial.seed(SCENE_SETS[version], live=live)
        serial_report = serial.scheduler.run_once()

        with ThreadDispatcher(3) as pool:
            threaded = make_enrichment_harness(dispatcher=pool)
            for version, live in registry.items():
                threaded.seed(SCENE_SETS[version], live=live)
            threaded_report = threaded.scheduler.run_once()

        assert threaded_report == serial_report
        assert threaded.store.snapshot() == serial.store.snapshot()
        assert sorted(threaded.writer.commands) == sorted(serial.writer.commands)
