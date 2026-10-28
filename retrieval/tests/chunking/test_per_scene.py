"""The per-scene chunker: one draft per scene with text."""

import dataclasses

from scene_rag.retrieval.chunking import models
from scene_rag.retrieval.chunking.config import ChunkingConfig, PerSceneConfig, UpstreamConfig
from scene_rag.retrieval.chunking.per_scene import PerSceneChunker
from tests.support.chunking_fakes import (
    DESCRIPTION_FP,
    TRANSCRIPT_FP,
    make_enrichment,
    make_record,
    make_records,
    make_scene,
)

RECORDS = make_records()


class TestGivenDefaultConfig:
    """Description then transcript, blank line between sections, space between segments."""

    def test_when_chunked_then_one_draft_per_scene_in_order(self) -> None:
        drafts = PerSceneChunker(ChunkingConfig()).chunk(RECORDS)

        assert [d.scene_indices for d in drafts] == [(0,), (1,), (2,)]
        assert [(d.start_seconds, d.end_seconds) for d in drafts] == [
            (0.0, 4.0),
            (4.0, 8.0),
            (8.0, 12.5),
        ]
        assert drafts[1].text == "description of scene 1\n\ntranscript of scene 1"
        assert drafts[1].enrichments == tuple(e.ref for e in RECORDS[1].enrichments)

    def test_when_segments_unordered_and_padded_then_sorted_and_stripped(self) -> None:
        scene = make_scene(0, (0.0, 4.0))
        transcript = make_enrichment(
            "sha256:aa",
            scene,
            "transcript",
            TRANSCRIPT_FP,
            segments=(
                models.SegmentView(2.0, 3.0, "  later "),
                models.SegmentView(0.5, 1.5, "first"),
                models.SegmentView(1.5, 2.0, "   "),
            ),
        )
        record = make_record(scene, (transcript,))

        (draft,) = PerSceneChunker(ChunkingConfig()).chunk([record])

        assert draft.text == "first later"
        assert draft.enrichments == (transcript.ref,)

    def test_when_scene_has_no_text_then_no_draft(self) -> None:
        scene = make_scene(0, (0.0, 4.0))
        empty = make_enrichment("sha256:aa", scene, "description", DESCRIPTION_FP, segments=())
        record = make_record(scene, (empty,))

        assert PerSceneChunker(ChunkingConfig()).chunk([record]) == ()

    def test_when_enricher_silent_then_not_among_used_enrichments(self) -> None:
        scene = make_scene(0, (0.0, 4.0))
        silent = make_enrichment("sha256:aa", scene, "transcript", TRANSCRIPT_FP, segments=())
        spoken = make_enrichment("sha256:aa", scene, "description", DESCRIPTION_FP)
        record = make_record(scene, (spoken, silent))

        (draft,) = PerSceneChunker(ChunkingConfig()).chunk([record])

        assert draft.text == "description of scene 0"
        assert draft.enrichments == (spoken.ref,)

    def test_when_unlisted_enricher_present_then_ignored(self) -> None:
        scene = make_scene(0, (0.0, 4.0))
        ocr = make_enrichment("sha256:aa", scene, "ocr", "fedcba987654")
        record = dataclasses.replace(RECORDS[0], enrichments=(*RECORDS[0].enrichments, ocr))

        (draft,) = PerSceneChunker(ChunkingConfig()).chunk([record])

        assert "ocr" not in draft.text
        assert ocr.ref not in draft.enrichments


class TestGivenCustomConfig:
    """Section order and separators follow the hyperparameters."""

    def test_when_order_and_separators_changed_then_text_follows(self) -> None:
        config = ChunkingConfig(
            per_scene=PerSceneConfig(
                enrichers=("transcript", "description"), section_separator=" | "
            )
        )

        drafts = PerSceneChunker(config).chunk(RECORDS)

        assert drafts[0].text == "transcript of scene 0 | description of scene 0"

    def test_when_only_one_enricher_listed_then_only_its_text(self) -> None:
        config = ChunkingConfig(
            upstream=UpstreamConfig(required_enrichers=("transcript",)),
            per_scene=PerSceneConfig(enrichers=("transcript",)),
        )

        drafts = PerSceneChunker(config).chunk(RECORDS)

        assert drafts[2].text == "transcript of scene 2"

    def test_when_named_then_per_scene_with_twelve_digit_fingerprint(self) -> None:
        chunker = PerSceneChunker(ChunkingConfig())

        assert chunker.name == "per_scene"
        assert len(chunker.fingerprint) == 12
