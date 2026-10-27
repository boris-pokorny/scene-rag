"""Invariants of content addressing and reconciliation, checked over generated histories."""

from datetime import timedelta

from hypothesis import given, settings
from hypothesis import strategies as st
from support.fakes import T0, FakeMedia, make_harness, sha256_version

ASSET_IDS = [f"asset-{index:016x}" for index in range(4)]
BLOBS = [b"a", b"bb", b"ccc"]

# A listing maps asset ids to the bytes the source holds for them.
listings = st.dictionaries(st.sampled_from(ASSET_IDS), st.sampled_from(BLOBS))
histories = st.lists(listings, min_size=1, max_size=6)


def media_for(listing: dict[str, bytes]) -> list[FakeMedia]:
    # Same bytes imply the same mtime, so unchanged items take the no-hash path and changed
    # items take the hashing path, as they would on a real filesystem.
    return [
        FakeMedia(asset_id, blob, modified_at=T0 + timedelta(seconds=len(blob)))
        for asset_id, blob in sorted(listing.items())
    ]


class TestGivenAnyListing:
    """Properties over generated listings."""

    @given(listing=listings)
    @settings(max_examples=60)
    def test_when_ingested_then_one_version_per_distinct_bytes(
        self, listing: dict[str, bytes]
    ) -> None:
        harness = make_harness(*media_for(listing))

        harness.service.run_once()

        live = harness.registry.list_by_source("local")
        assert {record.version for record in live} == {sha256_version(b) for b in listing.values()}
        for record in live:
            expected = {
                aid for aid, blob in listing.items() if sha256_version(blob) == record.version
            }
            assert {reference.asset_id for reference in record.sources} == expected


class TestGivenAnyHistory:
    """Properties over generated sequences of listings."""

    @given(history=histories)
    @settings(max_examples=60)
    def test_when_ingested_then_live_state_matches_last_listing(
        self, history: list[dict[str, bytes]]
    ) -> None:
        harness = make_harness()

        for listing in history:
            harness.clock.tick()
            harness.source.items = media_for(listing)
            report = harness.service.run_once()
            assert report.failed == 0

        final = history[-1]
        held = {
            reference.asset_id: record.version
            for record in harness.registry.list_by_source("local")
            for reference in record.sources
        }
        assert held == {aid: sha256_version(blob) for aid, blob in final.items()}
        for version, record in harness.registry.snapshot().items():
            if version not in held.values():
                assert record.retired_at is not None and record.sources == ()
