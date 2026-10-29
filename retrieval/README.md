# scene-rag-retrieval

The `retrieval` service of scene-rag: steps 4–5 of the indexing pipeline and the
`query-api` role.

- **step 4, `chunking`**: cut every scene set complete under the accepted upstream
  fingerprints into chunks (per scene by default) and write them, with one chunk set per
  scene set and fingerprint, to the Chunk Store.
- **step 5, `embedding`**: embed every chunk set under the accepted chunking fingerprint
  (`embeddinggemma` on Ollama by default) into the Search Index (Qdrant, one collection per
  embedding fingerprint), and record one index set per chunk set in the index ledger.
- **`query-api`**: **search** (the scene spans relevant to a question that the caller may
  see, best first) and **answer** (a cited answer over them, or a refusal), over HTTP
  (`POST /search`, `POST /answer`) and MCP (`/mcp` on the same port, or stdio).

Media data is read only through extraction's `scene-api`, never from its stores. Steps never
import each other; each reads the previous step's collections through its own read model,
pinned by `contracts/internal/`.

## Setup

pdm in PEP 582 mode (`use_venv = false`): packages land in `__pypackages__/`.

```sh
pdm install
cp .env.example .env          # local settings, loaded by `pdm run`; ignored by git
docker run -d --name scene-rag-mongo -p 27017:27017 docker.io/mongo
docker run -d --name scene-rag-qdrant -p 6333:6333 docker.io/qdrant/qdrant:v1.20.0
docker run -d --name scene-rag-ollama -p 11434:11434 docker.io/ollama/ollama:0.40.12
pdm run pull-models           # the embedding model and the generator, then exit
```

Chunking also needs a `scene-api` over filled stores (`pdm run scene-api` in `../extraction`),
or use [Docker Compose](#run-with-docker-compose) for everything at once.

## Run

```sh
pdm run worker --step chunking --once     # one pass; without --once, poll in a loop
pdm run worker --step embedding --once
pdm run query-api                         # HTTP and MCP on QUERY_API_HOST:QUERY_API_PORT
pdm run query-api --transport stdio --entitlements public   # MCP for a local client
```

Settings (where and how fast) come from the environment. Hyperparameters (what is produced)
are the defaults of each config dataclass (`ChunkingConfig`, `EmbeddingConfig`,
`QueryConfig`) and change only through a pull request, which changes the fingerprint or the
`query_config` id. Every backend URL is probed at startup; a missing required variable or an
unreachable backend ends the process with a message naming it.

All variables are prefixed `SCENE_RAG_RETRIEVAL_`; sample values are in `.env.example`.

| Variable | Meaning (default) | Read by |
|---|---|---|
| `SCENE_API_URL` | Base URL of extraction's scene-api. | chunking, query-api |
| `CHUNK_STORE_URI` | MongoDB connection string of the Chunk Store and the index ledger; credentials are redacted from logs. | all |
| `CHUNK_STORE_DATABASE` | Database of the `chunks`, `chunk_sets`, `index_sets` and `indexes` collections. | all |
| `SEARCH_INDEX_URL` | Base URL of the Qdrant server. | embedding, query-api |
| `POLL_INTERVAL_SECONDS` | Pause between passes, and after a failed one. | worker loop only |
| `OLLAMA_URL` | Ollama serving the models (`http://localhost:11434`). | optional |
| `SCENE_API_TIMEOUT_SECONDS`, `SEARCH_INDEX_TIMEOUT_SECONDS` | Longest wait for one answer (`30`). | optional |
| `OLLAMA_TIMEOUT_SECONDS` | Longest wait for one embedding batch or query (`120`). | optional |
| `CHUNKING_WORKERS`, `EMBEDDING_WORKERS` | Sets processed at once within a pass (`1`). | optional |
| `EMBEDDING_BATCH_SIZE` | Texts per Ollama request (`32`); lower it on a slow CPU rather than raising the timeout. | optional |
| `QUERY_API_HOST`, `QUERY_API_PORT` | Where query-api listens (`127.0.0.1`, `8081`; `0` picks a port). | optional |
| `QUERY_API_ALLOWED_HOSTS` | Host header values served, comma-separated, without the port; others get 400, against DNS rebinding. `*` serves every host, for a gateway that checks it (`127.0.0.1,localhost`). | optional |
| `QUERY_API_GENERATOR_TIMEOUT_SECONDS` | Longest wait for one generated answer (`300`). | optional |

### Run with Docker Compose

`compose.yaml` at the repository root wires a complete instance: extraction (MongoDB,
Ollama, its workers, `scene-api`), Qdrant, and this service's workers and `query-api`, built
from `Dockerfile` here. A one-shot `ollama-pull-retrieval` job runs `pull-models` first.
MongoDB, Ollama, Qdrant and query-api are published on localhost only; query-api because
nothing asserts the entitlements header yet. `CHUNKING_WORKERS`, `EMBEDDING_WORKERS`,
`EMBEDDING_BATCH_SIZE`, `POLL_INTERVAL_SECONDS` and the `*_PORT` variables can be overridden
from the shell or a `.env` next to `compose.yaml`.

```sh
cd ..                                         # the repository root
docker compose up --build -d
docker compose logs -f chunking embedding
docker compose run --rm chunking worker --step chunking --once
mongosh scene_rag_retrieval --eval 'db.chunk_sets.countDocuments(); db.indexes.findOne()'
curl -s localhost:6333/collections/chunks_<fingerprint> | jq .result.points_count   # = chunk count
curl -s localhost:8081/health | jq                                                  # what is served
curl -s -X POST localhost:8081/answer -H 'Content-Type: application/json' \
  -H 'X-Scene-Rag-Entitlements: public' -d '{"question": "what is shown?", "top_k": 3}' | jq
```

## How it works

Both steps derive pending work instead of flagging it: a pass lists the upstream sets,
subtracts those with a completion marker (chunk set, index set) under the step's fingerprint,
and hands the rest to a `Dispatcher`. A set ends *chunked*/*indexed*, *deferred* (upstream
still changing), *duplicate* (another worker finished first), *failed*, or, in chunking only,
*dropped* (asset retired). A
fingerprint covers the config and the output-defining modules only, so a changed timeout,
log line or worker count reprocesses nothing.

### Chunking

Per scene set:

1. **Pins.** For each enricher in `UpstreamConfig.required_enrichers` (description and
   transcript by default): the pin `enrichment_pins` names, else the greatest fingerprint
   complete over the set. A set lacking a required enricher waits. A new Media model thus
   reruns chunking under a new fingerprint, next to the old chunks.
2. **Records.** Fetched with those pins, paged to the end. A set that changed since listing
   is deferred; a retired or unknown asset version drops it.
3. **Chunks.** The `Chunker` drafts them; the per-scene one joins each scene's enricher text
   in `PerSceneConfig.enrichers` order, and skips scenes without text. The service adds the
   id `hash(scene record versions + fingerprint + index)`, the asset's sources and
   entitlements, the keyframes, and the lineage of every record read.
4. **Chunk set.** After a final liveness check, the chunks and then the chunk set (id
   `hash(scene set version + fingerprint)`, zero chunks is valid) are inserted. Chunks left
   by a worker that died are found as duplicates by the rerun.

Fingerprinted: `ChunkingConfig` (pins included), `models.py`, `fingerprint.py`,
`per_scene.py`. Schemas: `contracts/internal/chunk.schema.json`, `chunk-set.schema.json`.
`scene_api_client.py` is this service's decoder of extraction's published contracts.

### Embedding

Before the first pass, the collection `chunks_<fingerprint>` is created with the model's
dimensions and the configured distance (a mismatching existing one ends the process), with
payload indexes on `entitlements`, `asset_version` and `chunk_set_version`. One `indexes`
document per fingerprint names the collection, model and digest, dimensions and query
prefix: query-api reads it to embed queries exactly as the index was built. Per chunk set:

1. **Chunks.** Read in index order; a set not yet complete is deferred.
2. **Vectors.** Embedded in batches (`ModelConfig.document_prefix` in front, truncated at
   `num_ctx` tokens) and upserted idempotently: id = UUID5 of the chunk version, the index
   entry as payload.
3. **Index set.** One `index_sets` document (id `hash(chunk set version + fingerprint)`)
   with the vector count and every chunk version read.

Fingerprinted: `EmbeddingConfig`, `models.py`, `fingerprint.py`, `ollama_embedding.py`, and
the model digest Ollama reports, so a re-pulled model with other bytes is a new index.
Schemas: `contracts/internal/index-entry.schema.json`, `index-set.schema.json`,
`index.schema.json`. Another vector database or model gateway is a module behind
`SearchIndexReader`/`SearchIndexWriter` or `EmbeddingModel`, composed in `bootstrap.py`.

### Query-api

The role serves one index: the one `UpstreamConfig.embedding_fingerprint` pins, else the
newest in the ledger. At startup it checks that Ollama holds the index's model under the
recorded digest, that the collection has the recorded dimensions, and that a probe query
embeds to them; then it resolves the generator (`GeneratorConfig.name`, `qwen2.5:3b`).

Per request, over either adapter:

1. **Caller.** Entitlements come from the `X-Scene-Rag-Entitlements` header (trusted; a
   gateway is expected to assert it) or, over stdio, `--entitlements`. Without any, nothing
   is visible, keyframes included, and no model is called. A request to a host outside
   `QUERY_API_ALLOWED_HOSTS` is refused first.
2. **Request.** `question` non-blank, at most `RetrieverConfig.max_question_chars`; `top_k`
   at least one, defaulting to `top_k` and capped by `max_top_k`; `filters.asset_version`
   optional.
3. **Search.** The question is embedded behind the index's query prefix; the Search Index
   returns the closest points sharing an entitlement with the caller, filtered before
   ranking. A source carries versions, scene indices, time span, score, text and source
   references, but not its entitlements, which would name groups the caller is not in.
4. **Answer.** The `Reranker` (none) orders the sources; the `RefusalPolicy` refuses, without
   calling the generator, when no source reaches `RefusalConfig.min_score`; otherwise the
   `PromptConfig` prompt goes to the `AnswerGenerator` and its `[n]` markers are checked
   against the sources (`citations`, `invalid_citations`).
5. **Served.** Every answer names the `query_config` id (config values, the modules
   `models.py`, `fingerprint.py`, `retriever.py`, `reranker.py`, `refusal.py`, `prompt.py`,
   `citations.py`, `ollama.py`, and the generator digest), the embedding and chunking
   fingerprints, and the collection, so an evaluation records exactly what it measured.

MCP (official SDK) exposes the tools `search_scenes` (primary: the client's model writes the
answer) and `answer`, returning the published documents as structured output, and the
resources `scene://scene-sets/{scene_set_version}/{index}` (for a caller entitled to the
asset) and `keyframe://derivatives/{digest}`, proxied from scene-api. Over HTTP it is a
stateless route of the same application. Errors carry the HTTP error code; a failing backend
is reported by code and a fixed message, its detail goes to the log.

Published contracts (`contracts/published/`): the schemas with samples,
`query-api.openapi.json` (`pdm run openapi`) and `query-api.mcp.json`
(`pdm run mcp-schemas`); contract tests fail when a committed copy is stale. The role never
writes a store. Retriever, reranker, refusal policy, generator and scene reader are
extension points composed in `bootstrap.py`.

## Deferred

Liveness checks and the purge of retired assets' points (`RetiredFilter` is a null object
until `AssetRetired` is consumed), EmbeddingGemma's shorter vector sizes, hybrid retrieval, a
reranker, release manifests, authentication (the header is trusted), and matching keyframes
against the caller's entitlements (any entitled caller who knows a digest gets the JPEG).

## Develop

```sh
pdm run lint              # ruff check + ruff format --check
pdm run format            # ruff format
pdm run typecheck         # mypy --strict
pdm run test              # pytest, coverage threshold 95%; MongoDB and Qdrant cases skip when unreachable
pdm run test-integration  # only those cases, without the threshold
```

The integration cases need a MongoDB at `CHUNK_STORE_URI` and, when marked `qdrant`, a Qdrant
at `SEARCH_INDEX_URL` (`docker compose up -d mongo qdrant` at the root serves both).

Tests are named `given_<state>_when_<event>_then_<outcome>`, one class per subject. Fakes live
in `tests/support/{chunking,embedding,query}_fakes.py` and are wired into the
`chunking_harness`, `embedding_harness` and `query_harness` fixtures in `tests/conftest.py`;
a class may override a fixture to seed state. The Search Index cases run against both the
fakes and Qdrant, and `tests/query_api/test_parity.py` checks that HTTP and MCP answer alike.
Docstrings follow the Google style (ruff `D`), typing is strict (`ANN`, mypy strict).
`tests/architecture/` enforces the import rules: no cross-step imports, nothing from
`scene_rag.extraction`, the environment read only in `bootstrap.py`, pymongo, HTTP, FastAPI
and the MCP SDK each confined to their modules, and no write verb in the query-api role.
