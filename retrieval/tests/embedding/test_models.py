"""Identity derivations of the step's records."""

import uuid
from datetime import UTC, datetime

from scene_rag.retrieval.embedding import models


class TestGivenVersions:
    """Content-hash identities."""

    def test_when_entry_versioned_then_hash_of_chunk_and_fingerprint(self) -> None:
        version = models.index_entry_version("sha256:ab", "0123456789ab", "sha256")

        assert version.startswith("sha256:") and len(version) == 7 + 64
        assert version == models.index_entry_version("sha256:ab", "0123456789ab", "sha256")
        assert version != models.index_entry_version("sha256:ab", "ba9876543210", "sha256")
        assert version != models.index_entry_version("sha256:ac", "0123456789ab", "sha256")

    def test_when_set_versioned_then_hash_of_set_and_fingerprint(self) -> None:
        version = models.index_set_version("sha256:ab", "0123456789ab", "sha512")

        assert version.startswith("sha512:")
        assert version != models.index_set_version("sha256:ab", "ba9876543210", "sha512")


class TestGivenPointIds:
    """UUIDs derived from chunk versions."""

    def test_when_derived_then_stable_uuid5_of_the_chunk_version(self) -> None:
        point_id = models.point_id("sha256:ab")

        assert uuid.UUID(point_id).version == 5
        assert point_id == models.point_id("sha256:ab")
        assert point_id != models.point_id("sha256:ac")

    def test_when_collection_named_then_prefixed_fingerprint(self) -> None:
        assert models.collection_name("0123456789ab") == "chunks_0123456789ab"


class TestGivenTimestamps:
    """Millisecond truncation."""

    def test_when_truncated_then_microseconds_dropped(self) -> None:
        moment = datetime(2026, 10, 28, 9, 0, 0, 123456, tzinfo=UTC)

        assert models.truncate_to_millisecond(moment).microsecond == 123000
