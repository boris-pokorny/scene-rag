# Published contracts of `extraction`

Contracts other services consume, released to the schema registry:

| Contract | Consumers | Published by |
|---|---|---|
| scene record | `retrieval` (chunking) | enrichment, scene-catalog |
| `AssetVersioned`, `AssetRetired` events | `retrieval` | ingestion (outbox) |
| `ScenesPublished` event | `retrieval` | enrichment |
| `scene-catalog` API (OpenAPI) | `retrieval` | scene-catalog |

None exists yet. The Asset Registry is an internal store; its record schema lives in
`../internal/`.
