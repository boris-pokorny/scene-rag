"""The query-api role of the retrieval service: questions in, cited scene sources out.

The query-time path of the pipeline. The role serves two use cases, **search** (the
scenes relevant to a question) and **answer** (a cited answer written over them), through
two adapters with no logic of their own: HTTP (``POST /search``, ``POST /answer``) and MCP
(tools ``search_scenes`` and ``answer``, resources ``scene://`` and ``keyframe://``). It
reads the index ledger and the Search Index the embedding step wrote, through its own read
models pinned by ``contracts/internal/``, and scene records through the scene-api of
``extraction``. It never imports a step and never writes a store.
"""
