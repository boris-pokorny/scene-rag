"""Read model over the Enrichment Store.

`MongoEnrichmentReader` reads the ``enrichments`` collection enrichment writes. The document
shape is pinned by ``contracts/internal/enrichment.schema.json``; the role keeps its own view
rather than importing enrichment's model, so the two share a contract, not code.
"""

from collections.abc import Iterable, Sequence
from typing import Any, Final

from pymongo import ASCENDING, MongoClient

from scene_rag.extraction.scene_api import models

COLLECTION: Final = "enrichments"

# pymongo documents are untyped dictionaries; `Any` is confined to this boundary.
type Document = dict[str, Any]

_SCENE_ORDER: Final = [
    ("scene_index", ASCENDING),
    ("enricher", ASCENDING),
    ("lineage.fingerprint", ASCENDING),
]


def segment_from_document(document: Document) -> models.SemanticSegmentView:
    """Decode a semantic segment stored inside a record."""
    return models.SemanticSegmentView(
        start_seconds=document["start_seconds"],
        end_seconds=document["end_seconds"],
        text=document["text"],
        language=document["language"],
        confidence=document["confidence"],
    )


def enrichment_from_document(document: Document) -> models.EnrichmentView:
    """Decode an enrichment document into the role's view of it."""
    return models.EnrichmentView(
        version=document["_id"],
        scene_set_version=document["scene_set_version"],
        scene_index=document["scene_index"],
        enricher=document["enricher"],
        fingerprint=document["lineage"]["fingerprint"],
        classification=document["classification"],
        model=models.ModelRef(name=document["model"]["name"], digest=document["model"]["digest"]),
        segments=tuple(segment_from_document(segment) for segment in document["segments"]),
    )


def coverage_from_document(document: Document) -> models.Coverage:
    """Decode one group of the coverage aggregation."""
    key = document["_id"]
    return models.Coverage(
        scene_set_version=key["scene_set_version"],
        enricher=key["enricher"],
        fingerprint=key["fingerprint"],
        scene_count=document["scene_count"],
    )


class MongoEnrichmentReader:
    """`EnrichmentReader` over the enrichment collection; never writes to it."""

    def __init__(self, client: MongoClient[Document], database: str) -> None:
        """Bind to the enrichment collection in ``database`` on ``client``."""
        self._collection = client[database][COLLECTION]

    def list_coverage(self, scene_set_versions: Iterable[str]) -> Sequence[models.Coverage]:
        """Return, per scene set, enricher and fingerprint, how many scenes have a record.

        In no particular order: the groups come as the aggregation emits them, and the service
        sorts what it publishes.
        """
        pipeline: list[Document] = [
            {"$match": {"scene_set_version": {"$in": list(scene_set_versions)}}},
            {
                "$group": {
                    "_id": {
                        "scene_set_version": "$scene_set_version",
                        "enricher": "$enricher",
                        "fingerprint": "$lineage.fingerprint",
                    },
                    "scenes": {"$addToSet": "$scene_index"},
                }
            },
            {"$project": {"scene_count": {"$size": "$scenes"}}},
        ]
        return [
            coverage_from_document(document) for document in self._collection.aggregate(pipeline)
        ]

    def list_for_scenes(
        self,
        scene_set_version: str,
        scene_indices: Iterable[int],
        pins: Sequence[models.EnrichmentPin] = (),
    ) -> Sequence[models.EnrichmentView]:
        """Return the records of the given scenes, in (index, enricher, fingerprint) order.

        With pins, the (enricher, fingerprint) disjunction leads the criteria so the owner's
        compound index ``(enricher, lineage.fingerprint, scene_set_version, scene_index)``
        serves the whole query.
        """
        criteria: Document = {
            "scene_set_version": scene_set_version,
            "scene_index": {"$in": list(scene_indices)},
        }
        if pins:
            criteria["$or"] = [
                {"enricher": pin.enricher, "lineage.fingerprint": pin.fingerprint} for pin in pins
            ]
        cursor = self._collection.find(criteria).sort(_SCENE_ORDER)
        return [enrichment_from_document(document) for document in cursor]
