# Published contracts of `evaluation`

Contracts other services consume, released to the schema registry:

| Contract | Consumers | Published by | File |
|---|---|---|---|
| eval run verdict | `retrieval`'s release step (promotion gate) | compare | `eval-verdict.schema.json` |

The verdict sample under `samples/` is validated against the schema in CI, as is every
verdict `compare` writes in the tests.

What this service consumes, the query API, is published by `retrieval`. The service keeps
its own decoder of the responses in `src/scene_rag/evaluation/query_api_client.py`,
pinned by copies of retrieval's samples under `tests/contract/samples/`.

The dataset case and the stored eval run are internal: their schemas live in `../internal/`.
