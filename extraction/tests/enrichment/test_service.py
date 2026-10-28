"""The per-scene-set unit of work: pending scenes in, one record per scene and enricher out."""

import dataclasses
from collections.abc import Sequence

import pytest

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.protocols import VersionRetiredError
from scene_rag.extraction.enrichment.service import EnrichmentService
from tests.support.enrichment_fakes import (
    SPEECH_MODEL,
    T0,
    VISION_MODEL,
    EnrichmentHarness,
    FakeEnricher,
    FakeTranscriber,
    FakeVisionModel,
    RecordingEnrichmentWriter,
    make_scene_set,
    summary_of,
)

SCENE_SET = make_scene_set()
ALL = frozenset(range(3))


def pending(*names: str, indices: frozenset[int] = ALL) -> dict[str, frozenset[int]]:
    return dict.fromkeys(names or ("description", "transcript"), indices)


class TestGivenLiveSceneSet:
    """One scene set of a live version, nothing enriched yet."""

    @pytest.fixture
    def enrichment_harness(self, enrichment_harness: EnrichmentHarness) -> EnrichmentHarness:
        enrichment_harness.seed(SCENE_SET)
        return enrichment_harness

    def test_when_enriched_then_one_record_per_scene_and_enricher(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness

        outcome = harness.service.enrich(summary_of(SCENE_SET), pending())

        assert outcome == "enriched"
        records = harness.store.snapshot()
        assert len(records) == 6
        assert {(r.enricher, r.scene_index) for r in records.values()} == {
            (name, index) for name in ("description", "transcript") for index in range(3)
        }
        assert harness.writer.commands == [
            ("put", "description", 0),
            ("put", "description", 1),
            ("put", "description", 2),
            ("put", "transcript", 0),
            ("put", "transcript", 1),
            ("put", "transcript", 2),
        ]

    def test_when_enriched_then_record_carries_lineage_model_and_text(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness

        harness.service.enrich(summary_of(SCENE_SET), pending())

        fingerprints = harness.service.fingerprints
        version = models.enrichment_version(
            SCENE_SET.version, 1, "description", fingerprints["description"], "sha256"
        )
        record = harness.store.get(version)
        assert record is not None
        scene = SCENE_SET.scenes[1]
        assert record == models.Enrichment(
            version=version,
            scene_set_version=SCENE_SET.version,
            asset_version=SCENE_SET.asset_version,
            scene_index=1,
            enricher="description",
            classification="internal",
            model=VISION_MODEL,
            segments=(models.SemanticSegment(4.0, 8.0, "Seen: jpeg:1:0, jpeg:1:1"),),
            created_at=T0,
            lineage=models.Lineage(
                fingerprint=fingerprints["description"],
                inputs=(
                    f"scene-set:{SCENE_SET.version}",
                    f"derivative:{scene.keyframes[0].derivative.digest}",
                    f"derivative:{scene.keyframes[1].derivative.digest}",
                ),
                version=version,
            ),
        )
        transcript = harness.store.get(
            models.enrichment_version(
                SCENE_SET.version, 1, "transcript", fingerprints["transcript"], "sha256"
            )
        )
        assert transcript is not None
        assert transcript.model == SPEECH_MODEL
        assert transcript.segments == (models.SemanticSegment(4.5, 5.5, "Heard wav:1", "en", 0.9),)
        assert scene.audio_clip is not None
        assert transcript.lineage.inputs == (
            f"scene-set:{SCENE_SET.version}",
            f"derivative:{scene.audio_clip.digest}",
        )

    def test_when_only_some_scenes_pending_then_only_those_written(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness

        outcome = harness.service.enrich(
            summary_of(SCENE_SET),
            {"description": frozenset({2}), "transcript": frozenset({0, 2})},
        )

        assert outcome == "enriched"
        assert harness.writer.commands == [
            ("put", "description", 2),
            ("put", "transcript", 0),
            ("put", "transcript", 2),
        ]
        assert harness.vision.calls and len(harness.transcriber.calls) == 2

    def test_when_scene_index_out_of_range_then_skipped(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness

        outcome = harness.service.enrich(
            summary_of(SCENE_SET), {"description": frozenset({7}), "transcript": frozenset()}
        )

        assert outcome == "duplicate"
        assert harness.writer.commands == []

    def test_when_every_record_exists_then_duplicate_and_nothing_changed(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        harness.service.enrich(summary_of(SCENE_SET), pending())
        before = harness.store.snapshot()
        harness.clock.tick()

        outcome = harness.service.enrich(summary_of(SCENE_SET), pending())

        assert outcome == "duplicate"
        assert harness.store.snapshot() == before

    def test_when_scene_offers_nothing_then_empty_record_still_written(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        bare = make_scene_set(keyframes=0, audio=False)
        harness.seed(bare)

        outcome = harness.service.enrich(summary_of(bare), pending(indices=frozenset({0})))

        assert outcome == "enriched"
        written = [
            r for r in harness.store.snapshot().values() if r.scene_set_version == bare.version
        ]
        assert len(written) == 2
        assert all(
            r.segments == () and r.lineage.inputs == (f"scene-set:{bare.version}",) for r in written
        )

    def test_when_derivative_unavailable_then_scene_skipped_and_set_deferred(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        clip = SCENE_SET.scenes[1].audio_clip
        assert clip is not None
        del harness.derivatives.blobs[clip.digest]

        outcome = harness.service.enrich(summary_of(SCENE_SET), pending())

        assert outcome == "deferred"
        assert harness.writer.commands == [
            ("put", "description", 0),
            ("put", "description", 1),
            ("put", "description", 2),
            ("put", "transcript", 0),
            ("put", "transcript", 2),
        ]
        assert harness.store.list_done("transcript", harness.transcription.fingerprint) == {
            (SCENE_SET.version, 0),
            (SCENE_SET.version, 2),
        }

    def test_when_model_fails_then_scenes_skipped_and_set_failed(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        service = harness.variant_service(transcriber=FakeTranscriber(fail=True))

        outcome = service.enrich(summary_of(SCENE_SET), pending())

        assert outcome == "failed"
        assert [c[1] for c in harness.writer.commands] == ["description"] * 3

    def test_when_failure_and_deferral_mixed_then_failed_wins(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        clip = SCENE_SET.scenes[0].audio_clip
        assert clip is not None
        del harness.derivatives.blobs[clip.digest]
        service = harness.variant_service(vision=FakeVisionModel(fail=True))

        outcome = service.enrich(summary_of(SCENE_SET), pending())

        assert outcome == "failed"
        assert harness.writer.commands == [("put", "transcript", 1), ("put", "transcript", 2)]

    def test_when_writer_fails_then_later_scenes_still_attempted_and_set_failed(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        writer = RecordingEnrichmentWriter(harness.store, fail_after=2)
        service = harness.variant_service(writer=writer)

        outcome = service.enrich(summary_of(SCENE_SET), pending())

        assert outcome == "failed"
        assert len(writer.commands) == 6
        assert len(harness.store.snapshot()) == 2

    def test_when_scene_numbered_unlike_its_position_then_skipped(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        scenes = list(SCENE_SET.scenes)
        scenes[1] = dataclasses.replace(scenes[1], index=5)
        crooked = dataclasses.replace(SCENE_SET, scenes=tuple(scenes))
        harness.scenes.scene_sets[crooked.version] = crooked

        outcome = harness.service.enrich(summary_of(crooked), pending("description"))

        assert outcome == "enriched"
        assert harness.writer.commands == [("put", "description", 0), ("put", "description", 2)]

    def test_when_retired_before_start_then_version_retired_without_inference(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        harness.assets.retire(SCENE_SET.asset_version)

        with pytest.raises(VersionRetiredError, match=SCENE_SET.asset_version):
            harness.service.enrich(summary_of(SCENE_SET), pending())

        assert harness.vision.calls == [] and harness.transcriber.calls == []

    def test_when_retired_meanwhile_then_version_retired_before_write(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness

        def retire_then_answer(images: Sequence[bytes], prompt: str) -> str:
            harness.assets.retire(SCENE_SET.asset_version)
            return "late"

        retiring = FakeVisionModel(retire_then_answer)
        service = harness.variant_service(vision=retiring)

        with pytest.raises(VersionRetiredError, match=SCENE_SET.asset_version):
            service.enrich(summary_of(SCENE_SET), pending())

        assert harness.writer.commands == []

    def test_when_scene_set_gone_then_version_retired(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        del harness.scenes.scene_sets[SCENE_SET.version]

        with pytest.raises(VersionRetiredError, match="gone"):
            harness.service.enrich(summary_of(SCENE_SET), pending())

    def test_when_asset_unknown_then_version_retired(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        del harness.assets.records[SCENE_SET.asset_version]

        with pytest.raises(VersionRetiredError):
            harness.service.enrich(summary_of(SCENE_SET), pending())


class TestGivenEnrichers:
    """The sequence of enrichers a service is composed from."""

    def test_when_fingerprints_asked_then_one_per_enricher(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness

        assert harness.service.fingerprints == {
            "description": harness.description.fingerprint,
            "transcript": harness.transcription.fingerprint,
        }

    def test_when_names_collide_then_value_error(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        with pytest.raises(ValueError, match="unique"):
            enrichment_harness.variant_service(enrichers=(FakeEnricher("a"), FakeEnricher("a")))

    def test_when_no_enricher_then_value_error(self, enrichment_harness: EnrichmentHarness) -> None:
        with pytest.raises(ValueError, match="at least one"):
            enrichment_harness.variant_service(enrichers=())

    def test_when_custom_enricher_then_its_records_written(
        self, enrichment_harness: EnrichmentHarness
    ) -> None:
        harness = enrichment_harness
        harness.seed(SCENE_SET)
        service: EnrichmentService = harness.variant_service(
            enrichers=(FakeEnricher("ocr", "fedcba987654"),)
        )

        assert service.enrich(summary_of(SCENE_SET), pending("ocr")) == "enriched"
        assert {r.enricher for r in harness.store.snapshot().values()} == {"ocr"}
        assert harness.store.list_done("ocr", "fedcba987654") == {
            (SCENE_SET.version, i) for i in range(3)
        }
