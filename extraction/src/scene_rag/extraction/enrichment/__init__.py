"""Step 3, `enrichment`: describe, transcribe and otherwise annotate the scenes of a scene set.

The step reads scene sets (its own read model over the records segmentation writes), the
derivatives they reference and the Asset Registry (for liveness), runs every registered
`Enricher` over each scene, and writes one record per scene, enricher and fingerprint to the
Enrichment Store. It owns that store and is its only writer.
"""
