"""Step 4, `chunking`: turn scene records into retrieval units.

The step reads scene records through the `scene-api` of the extraction service (its own
read view over that published API; never a Media store), runs the registered `Chunker` over
every scene set that is complete under the accepted upstream fingerprints, and writes the
chunks and one chunk set per scene set and fingerprint to the Chunk Store. It owns that
store and is its only writer.
"""
