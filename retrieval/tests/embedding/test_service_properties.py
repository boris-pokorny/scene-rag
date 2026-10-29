"""Invariants of the pass over arbitrary Chunk Stores and partially indexed ledgers."""

from hypothesis import given, settings
from hypothesis import strategies as st

from scene_rag.retrieval.embedding.dispatch import ThreadDispatcher
from scene_rag.retrieval.embedding.report import EmbeddingReport
from tests.support.embedding_fakes import (
    make_chunk_set_view,
    make_chunk_views,
    make_embedding_harness,
    make_index_set,
    sha256_version,
)

VERSIONS = [sha256_version(bytes([i])) for i in range(4)]
CHUNK_SETS = {version: make_chunk_views(version, count=i % 3) for i, version in enumerate(VERSIONS)}

present = st.sets(st.sampled_from(VERSIONS))
done_before = st.sets(st.sampled_from(VERSIONS))


class TestGivenAnyChunkStore:
    """Chunk sets in any combination, some already indexed."""

    @given(stored=present, done=done_before)
    @settings(max_examples=40)
    def test_when_two_passes_run_then_exactly_one_index_set_per_chunk_set(
        self, stored: set[str], done: set[str]
    ) -> None:
        harness = make_embedding_harness()
        for version in stored:
            harness.reader.seed(
                CHUNK_SETS[version], make_chunk_set_view(CHUNK_SETS[version], asset_version=version)
            )
        fingerprint = harness.service.fingerprint
        for version in done & stored:
            harness.store.put_index_set(
                make_index_set(
                    CHUNK_SETS[version],
                    chunk_set=make_chunk_set_view(CHUNK_SETS[version], asset_version=version),
                    fingerprint=fingerprint,
                )
            )
        fresh = stored - done

        first = harness.scheduler.run_once()
        second = harness.scheduler.run_once()

        assert harness.store.list_done(fingerprint) == {
            make_chunk_set_view(CHUNK_SETS[v], asset_version=v).version for v in stored
        }
        assert first.indexed == len(fresh)
        assert first.failed == first.deferred == first.duplicate == 0
        assert second == EmbeddingReport()
        assert harness.search_index.count(harness.index.collection) == sum(
            len(CHUNK_SETS[v]) for v in fresh
        )

    @given(stored=present)
    @settings(max_examples=25, deadline=None)
    def test_when_threaded_then_same_records_as_serial(self, stored: set[str]) -> None:
        serial = make_embedding_harness()
        for version in stored:
            serial.reader.seed(
                CHUNK_SETS[version], make_chunk_set_view(CHUNK_SETS[version], asset_version=version)
            )
        serial_report = serial.scheduler.run_once()

        with ThreadDispatcher(3) as pool:
            threaded = make_embedding_harness(dispatcher=pool)
            for version in stored:
                threaded.reader.seed(
                    CHUNK_SETS[version],
                    make_chunk_set_view(CHUNK_SETS[version], asset_version=version),
                )
            threaded_report = threaded.scheduler.run_once()

        assert threaded_report == serial_report
        assert threaded.search_index.points(
            threaded.index.collection
        ) == serial.search_index.points(serial.index.collection)
        assert threaded.store.index_sets() == serial.store.index_sets()
