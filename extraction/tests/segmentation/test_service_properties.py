"""Invariants of derived pending work, checked over generated registry states."""

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.support.segmentation_fakes import (
    T0,
    make_segmentation_harness,
    make_view,
    sha256_version,
)

VERSIONS = [sha256_version(bytes([index])) for index in range(5)]

# A registry state maps versions to whether they are live.
registries = st.dictionaries(st.sampled_from(VERSIONS), st.booleans())


class TestGivenAnyRegistry:
    """Properties over generated registry states."""

    @given(registry=registries)
    @settings(max_examples=40)
    def test_when_passed_twice_then_one_set_per_live_version_and_second_pass_idle(
        self, registry: dict[str, bool]
    ) -> None:
        harness = make_segmentation_harness(
            *(make_view(v, retired_at=None if live else T0) for v, live in registry.items())
        )

        first = harness.service.run_once()
        second = harness.service.run_once()

        live = {v for v, is_live in registry.items() if is_live}
        assert first.segmented == len(live) and first.failed == 0
        assert second.segmented == 0 and len(harness.writer.commands) == len(live)
        assert harness.scene_store.list_asset_versions(harness.service.fingerprint) == live
        assert {s.asset_version for s in harness.scene_store.snapshot().values()} == live
