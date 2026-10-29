"""The embedding step of the retrieval service: chunks in, vectors in the Search Index out.

Step 5 of the indexing pipeline. The step reads the Chunk Store through its own read model
(`chunk_reader`, pinned by ``contracts/internal/chunk.schema.json``), embeds every chunk of
a chunk set with the registered `EmbeddingModel`, writes the points to the Search Index
behind `SearchIndexWriter` (one collection per embedding fingerprint) and records one index
set per chunk set in the index ledger. It never imports the chunking step.
"""
