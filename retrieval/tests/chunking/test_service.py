"""The per-scene-set unit of work: records in, chunks and a chunk set out."""

import dataclasses

import pytest

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import ChunkingConfig, UpstreamConfig
from scene_rag.retrieval.chunking.protocols import SceneApiError, VersionRetiredError
from tests.support.chunking_fakes import (
    DESCRIPTION_FP,
    T0,
    TRANSCRIPT_FP,
    ChunkingHarness,
    FakeChunker,
    RecordingChunkWriter,
    make_asset,
    make_enrichment,
    make_record,
    make_records,
    make_scene,
    make_source,
    sha256_version,
    summary_of,
)

RECORDS = make_records(
    asset=make_asset(
        make_source(),
        make_source("asset-0000000000000002", "library", "talk.mp4"),
        entitlements=frozenset({"public", "team-platform"}),
    )
)
SUMMARY = summary_of(RECORDS)


class TestGivenLiveSceneSet:
    """One complete scene set of a live version, nothing chunked yet."""

    @pytest.fixture
    def chunking_harness(self, chunking_harness: ChunkingHarness) -> ChunkingHarness:
        chunking_harness.seed(RECORDS)
        return chunking_harness

    def test_when_chunked_then_one_chunk_per_scene_then_chunk_set(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness

        outcome = harness.service.chunk(SUMMARY)

        assert outcome == "chunked"
        assert harness.writer.commands == [
            ("put_chunk", SUMMARY.version, 0),
            ("put_chunk", SUMMARY.version, 1),
            ("put_chunk", SUMMARY.version, 2),
            ("put_chunk_set", SUMMARY.version, 3),
        ]
        assert harness.reader.fetches == [
            (
                SUMMARY.version,
                (
                    models.EnrichmentPin("description", DESCRIPTION_FP),
                    models.EnrichmentPin("transcript", TRANSCRIPT_FP),
                ),
            )
        ]

    def test_when_chunked_then_chunk_carries_identity_provenance_and_access(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        fingerprint = harness.service.fingerprint

        harness.service.chunk(SUMMARY)

        record = RECORDS[1]
        version = models.chunk_version([record.version], fingerprint, 1, "sha256")
        chunk = harness.store.get_chunk(version)
        assert chunk == models.Chunk(
            version=version,
            scene_set_version=SUMMARY.version,
            asset_version=SUMMARY.asset_version,
            segmentation_fingerprint=SUMMARY.segmentation_fingerprint,
            chunk_index=1,
            scene_indices=(1,),
            start_seconds=4.0,
            end_seconds=8.0,
            text="description of scene 1\n\ntranscript of scene 1",
            sources=record.asset.sources,
            entitlements=frozenset({"public", "team-platform"}),
            classifications=("internal",),
            keyframes=tuple(
                models.KeyframeRef(k.derivative.digest, k.seconds) for k in record.scene.keyframes
            ),
            enrichments=tuple(e.ref for e in record.enrichments),
            created_at=T0,
            lineage=models.Lineage(
                fingerprint=fingerprint,
                inputs=(f"scene-record:{record.version}",),
                version=version,
            ),
        )

    def test_when_chunked_then_chunk_set_names_pins_and_every_record(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        fingerprint = harness.service.fingerprint

        harness.service.chunk(SUMMARY)

        version = models.chunk_set_version(SUMMARY.version, fingerprint, "sha256")
        assert harness.store.get_chunk_set(version) == models.ChunkSet(
            version=version,
            scene_set_version=SUMMARY.version,
            asset_version=SUMMARY.asset_version,
            segmentation_fingerprint=SUMMARY.segmentation_fingerprint,
            pins=(
                models.EnrichmentPin("description", DESCRIPTION_FP),
                models.EnrichmentPin("transcript", TRANSCRIPT_FP),
            ),
            chunk_count=3,
            created_at=T0,
            lineage=models.Lineage(
                fingerprint=fingerprint,
                inputs=tuple(f"scene-record:{r.version}" for r in RECORDS),
                version=version,
            ),
        )
        assert harness.store.list_done(fingerprint) == {SUMMARY.version}

    def test_when_chunked_again_then_duplicate_and_nothing_changed(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.service.chunk(SUMMARY)
        before = (harness.store.chunks(), harness.store.chunk_sets())
        harness.clock.tick()

        outcome = harness.service.chunk(SUMMARY)

        assert outcome == "duplicate"
        assert (harness.store.chunks(), harness.store.chunk_sets()) == before

    def test_when_worker_died_after_chunks_then_rerun_completes_the_set(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        dying = RecordingChunkWriter(harness.store, fail_after=3)
        with pytest.raises(RuntimeError, match="put_chunk_set"):
            harness.variant_service(writer=dying).chunk(SUMMARY)
        assert harness.store.chunk_sets() == {}

        outcome = harness.service.chunk(SUMMARY)

        assert outcome == "chunked"
        assert len(harness.store.chunks()) == 3 and len(harness.store.chunk_sets()) == 1

    def test_when_no_scene_has_text_then_empty_chunk_set_still_written(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        scene = make_scene(0, (0.0, 4.0))
        silent = [
            make_record(
                scene,
                (
                    make_enrichment("x", scene, "description", DESCRIPTION_FP, segments=()),
                    make_enrichment("x", scene, "transcript", TRANSCRIPT_FP, segments=()),
                ),
                asset_version=sha256_version(b"silent"),
            )
        ]
        harness.seed(silent)

        outcome = harness.service.chunk(summary_of(silent))

        assert outcome == "chunked"
        assert harness.store.chunks() == {}
        (chunk_set,) = harness.store.chunk_sets().values()
        assert chunk_set.chunk_count == 0

    def test_when_chunker_fails_then_error_propagates_before_any_write(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        service = harness.variant_service(chunker=FakeChunker(fail=True))

        with pytest.raises(RuntimeError, match="chunker down"):
            service.chunk(SUMMARY)

        assert harness.writer.commands == []

    def test_when_custom_chunker_then_its_drafts_stored_under_its_fingerprint(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        service = harness.variant_service(chunker=FakeChunker("fedcba987654"))

        assert service.chunk(SUMMARY) == "chunked"
        assert {c.text for c in harness.store.chunks().values()} == {
            "fake chunk 0",
            "fake chunk 1",
            "fake chunk 2",
        }
        assert harness.store.list_done("fedcba987654") == {SUMMARY.version}


class TestGivenPins:
    """Which fingerprints the records are fetched under."""

    def test_when_two_fingerprints_complete_then_greatest_fetched(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        records = make_records(
            enrichers={"description": "bbbbbbbbbbbb", "transcript": TRANSCRIPT_FP}
        )
        older = make_records(enrichers={"description": "aaaaaaaaaaaa"})
        both = tuple(
            dataclasses.replace(new, enrichments=(*new.enrichments, *old.enrichments))
            for new, old in zip(records, older, strict=True)
        )
        harness.seed(both)

        assert harness.service.chunk(summary_of(both)) == "chunked"
        (fetch,) = harness.reader.fetches
        assert fetch[1][0] == models.EnrichmentPin("description", "bbbbbbbbbbbb")
        assert all("description of scene" in c.text for c in harness.store.chunks().values())

    def test_when_configured_pin_then_that_fingerprint_fetched(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.seed(RECORDS)
        config = ChunkingConfig(
            upstream=UpstreamConfig(enrichment_pins=(f"description:{DESCRIPTION_FP}",))
        )
        service = harness.variant_service(config=config)

        assert service.chunk(SUMMARY) == "chunked"
        assert harness.reader.fetches[0][1][0] == models.EnrichmentPin(
            "description", DESCRIPTION_FP
        )

    def test_when_summary_not_ready_then_deferred_without_fetch(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.seed(RECORDS)
        only_description = dataclasses.replace(
            SUMMARY,
            enrichments=tuple(c for c in SUMMARY.enrichments if c.enricher == "description"),
        )

        assert harness.service.chunk(only_description) == "deferred"
        assert harness.reader.fetches == [] and harness.reader.liveness_checks == []
        assert harness.writer.commands == []


class TestGivenChangedSceneSet:
    """The scene set differs from its listing by the time the records are fetched."""

    @pytest.fixture
    def chunking_harness(self, chunking_harness: ChunkingHarness) -> ChunkingHarness:
        chunking_harness.seed(RECORDS)
        return chunking_harness

    def test_when_record_lacks_a_pin_then_deferred(self, chunking_harness: ChunkingHarness) -> None:
        harness = chunking_harness
        stripped = list(RECORDS)
        stripped[1] = dataclasses.replace(
            RECORDS[1],
            enrichments=tuple(e for e in RECORDS[1].enrichments if e.enricher != "transcript"),
        )
        harness.reader.records[SUMMARY.version] = tuple(stripped)

        assert harness.service.chunk(SUMMARY) == "deferred"
        assert harness.writer.commands == []

    def test_when_fewer_records_than_scenes_then_deferred(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.reader.records[SUMMARY.version] = RECORDS[:2]

        assert harness.service.chunk(SUMMARY) == "deferred"
        assert harness.writer.commands == []

    def test_when_scene_set_gone_then_scene_api_error(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        del harness.reader.records[SUMMARY.version]

        with pytest.raises(SceneApiError, match="no scene-set"):
            harness.service.chunk(SUMMARY)


class TestGivenRetiredVersion:
    """Liveness checks before any inference and before any write."""

    @pytest.fixture
    def chunking_harness(self, chunking_harness: ChunkingHarness) -> ChunkingHarness:
        chunking_harness.seed(RECORDS)
        return chunking_harness

    def test_when_retired_before_start_then_version_retired_without_fetch(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.reader.retire(SUMMARY.asset_version)

        with pytest.raises(VersionRetiredError, match=SUMMARY.asset_version):
            harness.service.chunk(SUMMARY)

        assert harness.reader.fetches == []

    def test_when_asset_unknown_then_version_retired(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        del harness.reader.assets[SUMMARY.asset_version]

        with pytest.raises(VersionRetiredError):
            harness.service.chunk(SUMMARY)

    def test_when_records_say_retired_then_version_retired_before_write(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness
        harness.reader.records[SUMMARY.version] = tuple(
            dataclasses.replace(r, live=False) for r in RECORDS
        )

        with pytest.raises(VersionRetiredError):
            harness.service.chunk(SUMMARY)

        assert harness.writer.commands == []

    def test_when_retired_during_chunking_then_version_retired_before_write(
        self, chunking_harness: ChunkingHarness
    ) -> None:
        harness = chunking_harness

        class RetiringChunker(FakeChunker):
            def chunk(self, records: object) -> tuple[models.ChunkDraft, ...]:
                harness.reader.assets[SUMMARY.asset_version] = False
                return super().chunk(records)  # type: ignore[arg-type]

        with pytest.raises(VersionRetiredError):
            harness.variant_service(chunker=RetiringChunker()).chunk(SUMMARY)

        assert harness.writer.commands == []
