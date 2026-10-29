"""The query-api role's read models decode the embedding step's internal contracts.

The step and the role share no code: the samples under contracts/internal/ are the pin.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.index_ledger import index_from_document
from scene_rag.retrieval.query_api.search_index import entry_from_payload

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "internal"


def load(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((CONTRACTS / "samples" / name).read_text())
    return document


class TestGivenEmbeddingSamples:
    """The embedding step's own samples."""

    def test_when_index_decoded_then_view_says_how_to_embed_a_query(self) -> None:
        sample = load("index.json")
        document = {**sample, "created_at": datetime.fromisoformat(sample["created_at"])}

        index = index_from_document(document)

        assert index == models.IndexView(
            fingerprint="5e2d4c6b8a90",
            collection="chunks_5e2d4c6b8a90",
            model=models.ModelRef("embeddinggemma", sample["model"]["digest"]),
            dimensions=768,
            distance="cosine",
            chunking_fingerprint=None,
            query_prefix="task: search result | query: ",
            truncate=True,
            num_ctx=2048,
            created_at=document["created_at"],
        )
        assert index.created_at.tzinfo is not None

    def test_when_entry_decoded_then_view_carries_what_citing_needs(self) -> None:
        sample = load("index-entry.json")

        entry = entry_from_payload(sample)

        assert entry == models.IndexEntryView(
            chunk_version=sample["chunk_version"],
            scene_set_version=sample["scene_set_version"],
            asset_version=sample["asset_version"],
            chunking_fingerprint="c7a1e2f3b4d5",
            scene_indices=(0,),
            start_seconds=0.0,
            end_seconds=4.0,
            text=sample["text"],
            sources=(models.SourceRef("local", "asset-0000000000000001", "clip.mp4"),),
            entitlements=frozenset({"public", "team-platform"}),
        )

    @pytest.mark.parametrize("name", ["index.json", "index-entry.json"])
    def test_when_schema_version_differs_then_rejected(self, name: str) -> None:
        sample = load(name)
        sample["schema_version"] = 2
        if name == "index.json":
            sample["created_at"] = datetime.fromisoformat(sample["created_at"])
        decode = index_from_document if name == "index.json" else entry_from_payload

        with pytest.raises(ValueError, match="schema_version 2"):
            decode(sample)
