"""Counters of a chunking pass."""

from scene_rag.retrieval.chunking.report import ChunkingReport


class TestGivenReport:
    """Counting outcomes of a pass."""

    def test_when_outcomes_counted_then_each_field_tallied(self) -> None:
        report = ChunkingReport.counting(
            ["chunked", "chunked", "duplicate", "dropped", "deferred", "failed"]
        )

        assert report == ChunkingReport(chunked=2, duplicate=1, dropped=1, deferred=1, failed=1)
        assert str(report) == "chunked=2 duplicate=1 dropped=1 deferred=1 failed=1"

    def test_when_no_outcomes_then_zeroes(self) -> None:
        assert ChunkingReport.counting([]) == ChunkingReport()
