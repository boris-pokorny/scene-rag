# Published contracts of `retrieval`

Contracts other services consume, released to the schema registry:

| Contract | Consumers | Published by | File |
|---|---|---|---|
| query API (OpenAPI 3.1, MCP tool schemas) | `evaluation`, UIs, LLM clients | query-api | not yet |

The Chunk Store and the Search Index are internal stores; their record schemas live in
`../internal/`. What this service consumes (the scene record, the scene set summary, the
asset version and the `scene-api` API) is published by `extraction`; the chunking step keeps
its own decoder of those documents in `src/scene_rag/retrieval/chunking/scene_api_client.py`,
pinned by the samples under `tests/contract/samples/`.
