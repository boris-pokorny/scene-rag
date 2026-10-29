"""Counters of an embedding pass."""

from scene_rag.retrieval.embedding.report import EmbeddingReport


class TestGivenReport:
    """Counting outcomes of a pass."""

    def test_when_outcomes_counted_then_each_field_tallied(self) -> None:
        report = EmbeddingReport.counting(["indexed", "indexed", "duplicate", "deferred", "failed"])

        assert report == EmbeddingReport(indexed=2, duplicate=1, deferred=1, failed=1)
        assert str(report) == "indexed=2 duplicate=1 deferred=1 failed=1"

    def test_when_no_outcomes_then_zeroes(self) -> None:
        assert EmbeddingReport.counting([]) == EmbeddingReport()
