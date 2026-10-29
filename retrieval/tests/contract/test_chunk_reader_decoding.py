"""The embedding step's read model decodes the Chunk Store contracts of the chunking step.

The two steps share no code: the samples under contracts/internal/ are the pin.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.chunk_reader import chunk_from_document, chunk_set_from_document

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "internal"


def load(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((CONTRACTS / "samples" / name).read_text())
    document["created_at"] = datetime.fromisoformat(document["created_at"])
    return document


class TestGivenChunkSamples:
    """The chunking step's own samples."""

    def test_when_chunk_decoded_then_view_carries_what_embedding_needs(self) -> None:
        sample = load("chunk.json")

        chunk = chunk_from_document(sample)

        assert chunk == models.ChunkView(
            version=sample["_id"],
            scene_set_version=sample["scene_set_version"],
            asset_version=sample["asset_version"],
            segmentation_fingerprint=sample["segmentation_fingerprint"],
            chunking_fingerprint=sample["lineage"]["fingerprint"],
            chunk_index=0,
            scene_indices=(0,),
            start_seconds=0.0,
            end_seconds=13.347,
            text=sample["text"],
            sources=(
                models.SourceView("local", "asset-3fa85f64d1a2b3c4", "sample-30s.mp4"),
                models.SourceView("library", "asset-0123456789abcdef", "talk.mp4"),
            ),
            entitlements=frozenset({"public", "team-platform"}),
            classifications=("internal",),
        )

    def test_when_chunk_set_decoded_then_view_carries_count_and_fingerprint(self) -> None:
        sample = load("chunk-set.json")

        chunk_set = chunk_set_from_document(sample)

        assert chunk_set == models.ChunkSetView(
            version=sample["_id"],
            scene_set_version=sample["scene_set_version"],
            asset_version=sample["asset_version"],
            chunking_fingerprint=sample["lineage"]["fingerprint"],
            chunk_count=1,
            created_at=sample["created_at"],
        )
        assert chunk_set.created_at.tzinfo is not None

    @pytest.mark.parametrize("name", ["chunk.json", "chunk-set.json"])
    def test_when_schema_version_differs_then_rejected(self, name: str) -> None:
        sample = load(name)
        sample["schema_version"] = 2
        decode = chunk_from_document if name == "chunk.json" else chunk_set_from_document

        with pytest.raises(ValueError, match="schema_version 2"):
            decode(sample)
