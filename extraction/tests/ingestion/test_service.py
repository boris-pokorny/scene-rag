from datetime import datetime, timedelta

import pytest

from scene_rag.extraction.ingestion import models
from scene_rag.extraction.ingestion.config import IngestionConfig
from scene_rag.extraction.ingestion.models import IngestionReport
from tests.support.fakes import (
    AGED,
    PUBLIC,
    T0,
    FailingMediaSource,
    FakeMedia,
    Harness,
    RacingWriter,
    RecordingWriter,
    make_record,
    make_reference,
    sha256_version,
)

A = "asset-00000000000000aa"
B = "asset-00000000000000bb"
ITEM = FakeMedia(A, b"hello")
VERSION = sha256_version(b"hello")


class TestGivenEmptyRegistry:
    """No version registered before the pass."""

    def test_when_new_item_then_registers_version(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"hello", entitlements=frozenset({"team-x"}))]

        report = harness.service.run_once()

        assert report == IngestionReport(registered=1)
        record = harness.registry.get(VERSION)
        assert record is not None
        assert record.media_type == "video/mp4"
        assert record.size_bytes == 5
        assert record.first_seen_at == T0
        assert record.entitlements == frozenset({"team-x"})
        assert record.lineage.fingerprint == harness.service.fingerprint
        assert record.lineage.version == record.version
        (reference,) = record.sources
        assert reference.key == ("local", A)
        assert reference.uri == f"fake:///{A}/clip.mp4"
        assert reference.seen_at == T0

    def test_when_two_items_share_bytes_then_one_version_two_references(
        self, harness: Harness
    ) -> None:
        harness.source.items = [
            FakeMedia(A, b"same", entitlements=frozenset({"public"})),
            FakeMedia(B, b"same", filename="copy.mp4", entitlements=frozenset({"team-x"})),
        ]

        report = harness.service.run_once()

        assert report == IngestionReport(registered=2)
        record = harness.registry.get(sha256_version(b"same"))
        assert record is not None
        assert {s.asset_id for s in record.sources} == {A, B}
        assert record.entitlements == frozenset({"public", "team-x"})

    def test_when_item_unreadable_then_counts_failure_and_continues(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"a", fail_open=True), FakeMedia(B, b"b")]

        report = harness.service.run_once()

        assert report == IngestionReport(registered=1, failed=1)
        assert harness.registry.get(sha256_version(b"b")) is not None

    def test_when_registry_fails_then_counts_failure(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"a")]
        failing = RecordingWriter(harness.registry, fail_on="register_version")

        report = harness.variant(writer=failing).run_once()

        assert report == IngestionReport(failed=1)

    def test_when_sha1_configured_then_version_uses_sha1(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"a")]

        harness.variant(config=IngestionConfig(hash_algorithm="sha1")).run_once()

        (record,) = harness.registry.list_by_source("local")
        assert record.version == "sha1:86f7e437faa5a7fce15d1ddcb9eaeaea377667b8"

    def test_when_item_large_then_reads_bounded_chunks(self, harness: Harness) -> None:
        reads: list[int] = []
        harness.source.items = [FakeMedia(A, b"x" * (3 * 2**20 + 1), read_sizes=reads)]

        harness.service.run_once()

        assert len(reads) >= 4
        assert max(reads) <= 2**20

    def test_when_item_empty_then_quarantines_it(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b""), FakeMedia(B, b"b")]

        report = harness.service.run_once()

        assert report == IngestionReport(registered=1, quarantined=1)
        assert [r.version for r in harness.registry.list_by_source("local")] == [
            sha256_version(b"b")
        ]

    def test_when_register_race_lost_then_adds_reference_to_rival(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"a")]

        report = harness.variant(writer=RacingWriter(harness.registry)).run_once()

        assert report == IngestionReport(registered=1)
        record = harness.registry.get(sha256_version(b"a"))
        assert record is not None
        assert {s.source_id for s in record.sources} == {"rival", "local"}

    def test_when_listing_fails_then_error_propagates(self, harness: Harness) -> None:
        with pytest.raises(OSError, match="source down"):
            harness.variant(source=FailingMediaSource()).run_once()

    def test_when_item_ingested_then_timestamps_match_clock(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"a")]
        harness.clock.current = T0 + timedelta(days=1)

        harness.service.run_once()

        (record,) = harness.registry.list_by_source("local")
        assert record.first_seen_at == T0 + timedelta(days=1)


class TestGivenRegisteredItem:
    """`ITEM` registered by an earlier pass; the clock has moved on to `later`."""

    @pytest.fixture
    def harness(self, harness: Harness) -> Harness:
        harness.source.items = [ITEM]
        harness.service.run_once()
        harness.writer.commands.clear()
        return harness

    @pytest.fixture
    def later(self, harness: Harness) -> datetime:
        return harness.clock.tick()

    def test_when_listing_matches_then_skips_hashing(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"hello", fail_open=True)]

        report = harness.service.run_once()

        assert report == IngestionReport(unchanged=1)
        assert harness.writer.commands == []

    def test_when_only_mtime_changes_then_refreshes_reference(
        self, harness: Harness, later: datetime
    ) -> None:
        harness.source.items = [FakeMedia(A, b"hello", modified_at=later - AGED)]

        report = harness.service.run_once()

        assert report == IngestionReport(unchanged=1)
        record = harness.registry.get(VERSION)
        assert record is not None
        (reference,) = record.sources
        assert reference.modified_at == later - AGED
        assert reference.seen_at == later

    def test_when_only_entitlements_change_then_refreshes_without_hashing(
        self, harness: Harness, later: datetime
    ) -> None:
        harness.source.items = [
            FakeMedia(A, b"hello", entitlements=frozenset({"team-x"}), fail_open=True)
        ]

        report = harness.service.run_once()

        assert report == IngestionReport(unchanged=1)
        assert harness.writer.commands == [("add_source_reference", VERSION, A)]
        record = harness.registry.get(VERSION)
        assert record is not None
        assert record.entitlements == frozenset({"team-x"})
        (reference,) = record.sources
        assert reference.seen_at == later

    def test_when_only_filename_changes_then_refreshes_without_hashing(
        self, harness: Harness
    ) -> None:
        harness.source.items = [FakeMedia(A, b"hello", filename="renamed.mp4", fail_open=True)]

        report = harness.service.run_once()

        assert report == IngestionReport(unchanged=1)
        record = harness.registry.get(VERSION)
        assert record is not None
        assert [s.filename for s in record.sources] == ["renamed.mp4"]

    def test_when_retire_fails_then_counts_failure_and_continues(self, harness: Harness) -> None:
        harness.registry.register_version(make_record(sha256_version(b"other"), make_reference(B)))
        harness.source.items = []
        failing = RecordingWriter(harness.registry, fail_on="remove_source_reference")

        report = harness.variant(writer=failing).run_once()

        assert report == IngestionReport(failed=2)
        assert len(failing.commands) == 2
        assert len(harness.registry.list_by_source("local")) == 2

    def test_when_bytes_change_then_supersedes_old_version(
        self, harness: Harness, later: datetime
    ) -> None:
        harness.source.items = [FakeMedia(A, b"changed", modified_at=later - AGED)]

        report = harness.service.run_once()

        assert report == IngestionReport(registered=1, superseded=1)
        new = sha256_version(b"changed")
        assert harness.writer.commands == [
            ("register_version", new),
            ("remove_source_reference", VERSION, A),
        ]
        old_record = harness.registry.get(VERSION)
        assert old_record is not None
        assert old_record.sources == ()
        assert old_record.retired_at == later
        assert [r.version for r in harness.registry.list_by_source("local")] == [new]

    def test_when_missing_from_listing_then_retires_it(
        self, harness: Harness, later: datetime
    ) -> None:
        harness.source.items = []

        report = harness.service.run_once()

        assert report == IngestionReport(retired=1)
        record = harness.registry.get(VERSION)
        assert record is not None
        assert record.retired_at == later
        assert record.sources == ()

    def test_when_truncated_to_empty_then_quarantines_and_retires(
        self, harness: Harness, later: datetime
    ) -> None:
        harness.source.items = [FakeMedia(A, b"", modified_at=later)]

        report = harness.service.run_once()

        assert report == IngestionReport(quarantined=1, retired=1)
        record = harness.registry.get(VERSION)
        assert record is not None and record.retired_at == later

    def test_when_rerun_then_registry_state_identical(
        self, harness: Harness, later: datetime
    ) -> None:
        before = harness.registry.snapshot()

        for _ in range(3):
            harness.service.run_once()

        assert harness.registry.snapshot() == before
        assert harness.writer.commands == []

    def test_when_other_source_holds_version_then_its_reference_kept(
        self, harness: Harness
    ) -> None:
        foreign = make_reference("asset-00000000000000ff", source_id="library")
        harness.registry.add_source_reference(VERSION, foreign)
        harness.source.items = []

        report = harness.service.run_once()

        assert report == IngestionReport(retired=1)
        record = harness.registry.get(VERSION)
        assert record is not None
        assert record.sources == (foreign,)
        assert record.is_live


class TestGivenSharedVersion:
    """One version held by two items, `A` and `B`, of the same source."""

    @pytest.fixture
    def harness(self, harness: Harness) -> Harness:
        harness.source.items = [FakeMedia(A, b"same"), FakeMedia(B, b"same")]
        harness.service.run_once()
        return harness

    def test_when_one_reference_retires_then_version_stays_live(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"same")]

        report = harness.service.run_once()

        assert report == IngestionReport(unchanged=1, retired=1)
        record = harness.registry.get(sha256_version(b"same"))
        assert record is not None
        assert [s.asset_id for s in record.sources] == [A]
        assert record.retired_at is None


class TestGivenCrashedPass:
    """A worker died after writing the new version of `A` and before removing the old reference.

    Both versions hold the key when the next pass starts.
    """

    NEW = sha256_version(b"changed")

    @pytest.fixture
    def harness(self, harness: Harness) -> Harness:
        harness.source.items = [ITEM]
        harness.service.run_once()
        later = harness.clock.tick()
        changed_at = later - AGED
        stranded = models.AssetVersion(
            version=self.NEW,
            media_type="video/mp4",
            size_bytes=7,
            sources=(
                models.SourceReference(
                    "local", A, f"fake:///{A}/clip.mp4", "clip.mp4", 7, changed_at, PUBLIC, later
                ),
            ),
            first_seen_at=later,
            lineage=models.Lineage(fingerprint=harness.service.fingerprint, version=self.NEW),
        )
        assert harness.registry.register_version(stranded)
        assert len(harness.registry.list_by_source("local")) == 2
        harness.source.items = [FakeMedia(A, b"changed", modified_at=changed_at)]
        harness.writer.commands.clear()
        return harness

    def test_when_next_pass_runs_then_converges_on_new_version(self, harness: Harness) -> None:
        report = harness.service.run_once()

        assert report == IngestionReport(unchanged=1, superseded=1)
        assert [r.version for r in harness.registry.list_by_source("local")] == [self.NEW]
        old_record = harness.registry.get(VERSION)
        assert old_record is not None and old_record.retired_at == harness.clock.current
        assert harness.writer.commands == [
            ("register_version", self.NEW),
            ("add_source_reference", self.NEW, A),
            ("remove_source_reference", VERSION, A),
        ]


class TestGivenRetiredVersion:
    """`ITEM` registered, then gone from the listing, so its version is tombstoned."""

    @pytest.fixture
    def harness(self, harness: Harness) -> Harness:
        harness.source.items = [ITEM]
        harness.service.run_once()
        harness.source.items = []
        harness.service.run_once()
        return harness

    def test_when_item_reappears_then_unretires_it(self, harness: Harness) -> None:
        harness.source.items = [ITEM]

        report = harness.service.run_once()

        assert report == IngestionReport(registered=1)
        record = harness.registry.get(VERSION)
        assert record is not None
        assert record.is_live


class TestGivenYoungItem:
    """Items modified less than `min_age_seconds` before the pass: still being copied, maybe."""

    def test_when_new_then_deferred_and_nothing_written(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"hello", modified_at=T0)]

        report = harness.service.run_once()

        assert report == IngestionReport(deferred=1)
        assert harness.writer.commands == []

    def test_when_registered_item_changed_then_old_reference_kept(self, harness: Harness) -> None:
        harness.source.items = [ITEM]
        harness.service.run_once()
        later = harness.clock.tick()
        harness.source.items = [FakeMedia(A, b"changed", modified_at=later)]
        harness.writer.commands.clear()

        report = harness.service.run_once()

        assert report == IngestionReport(deferred=1)
        assert harness.writer.commands == []
        assert [r.version for r in harness.registry.list_by_source("local")] == [VERSION]

    def test_when_aged_then_next_pass_registers_it(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"hello", modified_at=T0)]
        harness.service.run_once()
        harness.clock.tick()

        report = harness.service.run_once()

        assert report == IngestionReport(registered=1)
        assert harness.registry.get(VERSION) is not None


class TestGivenChangingItem:
    """Items whose size or modification time move while they are hashed."""

    def test_when_grown_after_hashing_then_deferred(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"hello", refreshed=FakeMedia(A, b"hello world"))]

        report = harness.service.run_once()

        assert report == IngestionReport(deferred=1)
        assert harness.registry.get(VERSION) is None

    def test_when_touched_after_hashing_then_deferred(self, harness: Harness) -> None:
        harness.source.items = [
            FakeMedia(A, b"hello", refreshed=FakeMedia(A, b"hello", modified_at=T0))
        ]

        report = harness.service.run_once()

        assert report == IngestionReport(deferred=1)

    def test_when_stream_longer_than_listed_then_deferred(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"hello", declared_size=3)]

        report = harness.service.run_once()

        assert report == IngestionReport(deferred=1)
        assert harness.writer.commands == []

    def test_when_refresh_fails_then_counts_failure(self, harness: Harness) -> None:
        harness.source.items = [FakeMedia(A, b"hello", fail_refresh=True)]

        report = harness.service.run_once()

        assert report == IngestionReport(failed=1)


class TestGivenReport:
    """The pass counters."""

    def test_when_rendered_then_key_value_pairs(self) -> None:
        assert str(IngestionReport(registered=2, failed=1)) == (
            "registered=2 unchanged=0 superseded=0 retired=0 quarantined=0 deferred=0 failed=1"
        )
