"""Invariants of the pass over arbitrary scene-api states and partially chunked stores."""

from hypothesis import given, settings
from hypothesis import strategies as st

from scene_rag.retrieval.chunking.dispatch import ThreadDispatcher
from scene_rag.retrieval.chunking.report import ChunkingReport
from tests.support.chunking_fakes import (
    DESCRIPTION_FP,
    make_chunk_set,
    make_chunking_harness,
    make_records,
    sha256_version,
    summary_of,
)

VERSIONS = [sha256_version(bytes([i])) for i in range(4)]
SCENE_SETS = {
    version: make_records(
        version,
        bounds=((0.0, 1.0), (1.0, 2.0)) if i % 2 else ((0.0, 1.5),),
        enrichers=None if i < 3 else {"description": DESCRIPTION_FP},
    )
    for i, version in enumerate(VERSIONS)
}
READY = {v for i, v in enumerate(VERSIONS) if i < 3}

registries = st.dictionaries(st.sampled_from(VERSIONS), st.booleans())
done_before = st.sets(st.sampled_from(VERSIONS))


class TestGivenAnyRegistry:
    """Live and retired versions in any combination, some sets already chunked."""

    @given(registry=registries, done=done_before)
    @settings(max_examples=40)
    def test_when_two_passes_run_then_exactly_one_chunk_set_per_ready_live_set(
        self, registry: dict[str, bool], done: set[str]
    ) -> None:
        harness = make_chunking_harness()
        for version, is_live in registry.items():
            harness.seed(SCENE_SETS[version], live=is_live)
        fingerprint = harness.service.fingerprint
        for version in done:
            harness.store.put_chunk_set(make_chunk_set(SCENE_SETS[version], fingerprint))
        fresh = {v for v, is_live in registry.items() if is_live and v in READY and v not in done}
        expected = {summary_of(SCENE_SETS[v]).version for v in fresh | done}

        first = harness.scheduler.run_once()
        second = harness.scheduler.run_once()

        assert harness.store.list_done(fingerprint) == expected
        assert first.chunked == len(fresh)
        assert first.failed == first.dropped == first.deferred == first.duplicate == 0
        assert second == ChunkingReport()
        chunks = harness.store.chunks().values()
        assert len(chunks) == sum(len(SCENE_SETS[v]) for v in fresh)

    @given(registry=registries)
    @settings(max_examples=25, deadline=None)
    def test_when_threaded_then_same_records_as_serial(self, registry: dict[str, bool]) -> None:
        serial = make_chunking_harness()
        for version, live in registry.items():
            serial.seed(SCENE_SETS[version], live=live)
        serial_report = serial.scheduler.run_once()

        with ThreadDispatcher(3) as pool:
            threaded = make_chunking_harness(dispatcher=pool)
            for version, live in registry.items():
                threaded.seed(SCENE_SETS[version], live=live)
            threaded_report = threaded.scheduler.run_once()

        assert threaded_report == serial_report
        assert threaded.store.chunks() == serial.store.chunks()
        assert threaded.store.chunk_sets() == serial.store.chunk_sets()
