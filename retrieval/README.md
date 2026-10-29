# scene-rag-retrieval

The `retrieval` service of scene-rag (Retrieval context): steps 4–5 of the indexing pipeline
and the `query-api` role. Implemented so far:

- **step 4, `chunking`**: read scene records through the `scene-api` of the `extraction`
  service, cut every scene set that is complete under the accepted upstream fingerprints
  into chunks with the registered `Chunker` (per scene by default), and write the chunks and
  one chunk set per scene set and fingerprint to the Chunk Store.
- **step 5, `embedding`**: read every chunk set of the Chunk Store under the accepted
  chunking fingerprint, embed its chunks with the registered `EmbeddingModel`
  (`embeddinggemma` on Ollama by default), write the vectors with their index entries to the
  Search Index (Qdrant by default, one collection per embedding fingerprint) and one index
  set per chunk set to the index ledger.

The service reads Media data only through `scene-api`: it never opens an extraction store.
Steps never import each other: embedding reads the chunk collections through its own read
model, pinned by the contracts under `contracts/internal/`.

## Setup

pdm in PEP 582 mode (`pdm.toml` sets `use_venv = false`); packages land in `__pypackages__/`,
no virtualenv is created.

```sh
pdm install
cp .env.example .env                        # local settings; .env is ignored by git
```

The Chunk Store and the index ledger are MongoDB collections; the Search Index is a Qdrant
server; the embedding model is served by Ollama. Start them locally, and a `scene-api` over
filled stores (`pdm run scene-api` in `../extraction`, or [Run with Docker
Compose](#run-with-docker-compose) for everything at once):

```sh
docker run -d --name scene-rag-mongo -p 27017:27017 docker.io/mongo
docker run -d --name scene-rag-qdrant -p 6333:6333 docker.io/qdrant/qdrant:v1.20.0
docker run -d --name scene-rag-ollama -p 11434:11434 docker.io/ollama/ollama:0.40.12
pdm run pull-models                           # puts the embedding model on the Ollama server
```

## Run

Settings (where and how fast) come from the environment; `.env` (copied from
`.env.example`, never committed) holds local values and is loaded by `pdm run`.
Hyperparameters (what is produced) are the defaults of each step's config dataclass
(`ChunkingConfig` in `src/scene_rag/retrieval/chunking/config.py`, `EmbeddingConfig` in
`src/scene_rag/retrieval/embedding/config.py`) and change only through a pull request, which
changes the step's fingerprint.

```sh
pdm run worker --step chunking --once         # one pass; needs no poll interval
pdm run worker --step chunking                # poll every SCENE_RAG_RETRIEVAL_POLL_INTERVAL_SECONDS
pdm run worker --step embedding --once        # the same for the embedding step
pdm run pull-models                           # pull EmbeddingConfig's model into Ollama, then exit
```

Every variable not marked optional is required; a missing one ends the process with a message
naming it. Sample values for a local run are in `.env.example`.

| Variable | Meaning | Read by |
|---|---|---|
| `SCENE_RAG_RETRIEVAL_SCENE_API_URL` | Base URL of the extraction service's scene-api. Probed at startup; an unreachable API ends the process. | chunking |
| `SCENE_RAG_RETRIEVAL_SCENE_API_TIMEOUT_SECONDS` | Longest wait for one scene-api answer; `30` by default. | chunking (optional) |
| `SCENE_RAG_RETRIEVAL_CHUNK_STORE_URI` | MongoDB connection string of the Chunk Store and the index ledger. Credentials in it are redacted from logs. | chunking, embedding |
| `SCENE_RAG_RETRIEVAL_CHUNK_STORE_DATABASE` | Database holding the `chunks`, `chunk_sets`, `index_sets` and `indexes` collections. | chunking, embedding |
| `SCENE_RAG_RETRIEVAL_POLL_INTERVAL_SECONDS` | Seconds between two passes of the worker loop; positive. Also the first pause after a failed pass. | loop only, unused with `--once` |
| `SCENE_RAG_RETRIEVAL_CHUNKING_WORKERS` | Scene sets chunked at once within a pass; `1` by default. The work is HTTP and store round trips, so a few workers overlap them. | chunking (optional) |
| `SCENE_RAG_RETRIEVAL_SEARCH_INDEX_URL` | Base URL of the Qdrant server. Probed at startup; an unreachable server ends the process. | embedding |
| `SCENE_RAG_RETRIEVAL_SEARCH_INDEX_TIMEOUT_SECONDS` | Longest wait for one Qdrant answer; `30` by default. | embedding (optional) |
| `SCENE_RAG_RETRIEVAL_OLLAMA_URL` | Ollama server serving the embedding model; `http://localhost:11434` by default. Probed at startup, and the configured model must be present (`pdm run pull-models`). | embedding, pull-models (optional) |
| `SCENE_RAG_RETRIEVAL_OLLAMA_TIMEOUT_SECONDS` | Longest wait for one embedding answer, i.e. one batch; `120` by default. | embedding (optional) |
| `SCENE_RAG_RETRIEVAL_EMBEDDING_BATCH_SIZE` | Texts embedded per request to Ollama; `32` by default. Lower it on a slow CPU rather than raising the timeout. | embedding (optional) |
| `SCENE_RAG_RETRIEVAL_EMBEDDING_WORKERS` | Chunk sets embedded at once within a pass; `1` by default. | embedding (optional) |

### Run with Docker Compose

`compose.yaml` at the repository root wires a complete instance: the extraction service
(MongoDB, Ollama, the three workers, `scene-api`), a Qdrant server, and this service's
`chunking` and `embedding` workers, built from `Dockerfile` here. Chunking reads
`http://scene-api:8080` and writes the `scene_rag_retrieval` database on the same MongoDB;
embedding reads that database, embeds through `http://ollama:11434` and writes
`http://qdrant:6333`. A one-shot `ollama-pull-retrieval` job runs this image's `pull-models`
role first, so the embedding model is named only in `EmbeddingConfig`. `CHUNKING_WORKERS`
(1), `EMBEDDING_WORKERS` (1), `EMBEDDING_BATCH_SIZE` (32), `POLL_INTERVAL_SECONDS` (30),
`QDRANT_PORT` (6333) and `OLLAMA_PORT` (11434; both published on localhost only, so a
`pdm run` worker on the host reaches the same servers) can be overridden from the shell or a
`.env` next to `compose.yaml`.

```sh
cd ..                                         # the repository root
docker compose up --build -d
docker compose logs -f chunking embedding
docker compose run --rm chunking worker --step chunking --once   # one pass instead of the loop
docker compose run --rm embedding worker --step embedding --once
mongosh scene_rag_retrieval --eval 'db.chunk_sets.countDocuments(); db.chunks.findOne()'
mongosh scene_rag_retrieval --eval 'db.indexes.findOne(); db.index_sets.find({}, {chunk_set_version: 1, vector_count: 1})'
curl -s localhost:6333/collections/chunks_<fingerprint> | jq .result.points_count   # equals the chunk count
```

### Chunking

Pending work is derived, never flagged: a pass lists the scene sets of live asset versions
(`GET /scene-sets`, filtered by the configured segmentation fingerprint and enrichment pins),
subtracts those with a chunk set under the step's fingerprint, skips those not complete under
the required enrichers, and hands the rest to a `Dispatcher`. Per scene set:

1. **Pins.** For every enricher in `UpstreamConfig.required_enrichers` (description and
   transcript by default), the configured pin if `enrichment_pins` names one, else the
   fingerprint complete over the set (the greatest when several are). A set with a required
   enricher not complete waits for enrichment. Pins are hyperparameters, so a release pins
   the fingerprints it accepts and a new Media model reruns chunking under a new fingerprint,
   next to the old chunks (design §4).
2. **Records.** Fetched with those pins (`GET /scene-sets/{version}/scenes?enrichment=…`),
   paged to the end, so every record carries one fingerprint per required enricher. A set
   that changed since listing (fewer records than scenes, a record missing a pin) is
   *deferred*; a retired or unknown asset version (a record no longer live, or
   `/asset-versions/{version}`) *drops* it; any other error counts as *failed*.
3. **Chunks.** The `Chunker` drafts the retrieval units; the per-scene chunker makes one per
   scene from the enrichers' text in `PerSceneConfig.enrichers` order (segments sorted by
   time and joined by `segment_separator`, sections by `section_separator`), none for a scene
   without text. The service adds identity and provenance: id `hash(scene record versions +
   fingerprint + index)`, the asset's sources and entitlements, the covered keyframes, the
   enrichment records used, and a lineage naming every scene record read.
4. **Chunk set.** After a final liveness check, the chunks are inserted one by one, then one
   chunk set (id `hash(scene set version + fingerprint)`) naming the pins, the chunk count
   (zero is valid) and every scene record version. It is the completion marker the next pass
   derives pending work from: a worker that dies before it leaves chunks a rerun finds as
   *duplicates* before completing the set; a chunk set another worker wrote first is a
   *duplicate* too.

The fingerprint covers `ChunkingConfig` (pins included) and the output-defining modules
`models.py`, `fingerprint.py` and `per_scene.py`; everything else (client, transport, store,
`service.py`, `scheduler.py`, `report.py`, `dispatch.py`, `bootstrap.py`) sits outside it, so
a changed timeout, log line or worker count never reprocesses anything. Record schemas:
`contracts/internal/chunk.schema.json` and `chunk-set.schema.json`. The consumed documents are
extraction's published contracts; `scene_api_client.py` is this service's copy of their
decoder, pinned by `tests/contract/`.

### Embedding

Pending work is derived the same way: a pass lists the chunk sets under
`UpstreamConfig.chunking_fingerprint` (every chunking fingerprint when `None`), subtracts
those with an index set under the step's fingerprint, and hands the rest to a `Dispatcher`.
Before the first pass the index of the fingerprint is prepared once: the Qdrant collection
`chunks_<fingerprint>` is created with the model's dimensions and the configured distance
(or verified when it exists; a different one ends the process), with keyword payload indexes
on `entitlements`, `asset_version` and `chunk_set_version`, and one document per fingerprint
is written to `indexes` naming the collection, the model and its digest, the dimensions and
the query prefix. That document is what `query-api` will read to search the right collection
and embed queries the way the index was built (design §3). Per chunk set:

1. **Chunks.** Fetched from `chunks` in index order under the set's chunking fingerprint. A
   set whose chunks are not all there is *deferred*; any error counts as *failed*.
2. **Vectors.** In batches of `SCENE_RAG_RETRIEVAL_EMBEDDING_BATCH_SIZE`, the `EmbeddingModel`
   embeds the chunk texts (`ModelConfig.document_prefix` in front, truncated at
   `ModelConfig.num_ctx` tokens) and the points are upserted: id = UUID (version 5) of the
   chunk version, vector, and the index entry as payload (chunk and set versions, asset
   version, fingerprints, scene indices, time span, entitlements, sources, text, lineage).
   Upserts are idempotent, so a worker that dies halfway leaves points a rerun overwrites.
3. **Index set.** One document in `index_sets` (id `hash(chunk set version + fingerprint)`)
   naming the collection, the vector count (zero is valid) and every chunk version read: the
   completion marker the next pass derives pending work from; one another worker wrote first
   is a *duplicate*.

The fingerprint covers `EmbeddingConfig` (chunking pin, model name, prefixes, truncation,
context, distance), the output-defining modules `models.py`, `fingerprint.py` and
`ollama_embedding.py`, and the digest of the model as Ollama reports it, so a re-pulled
model with other bytes is a new index. Everything else (`chunk_reader.py`, `index_store.py`,
`qdrant_index.py`, `http_transport.py`, `index_setup.py`, `service.py`, `scheduler.py`,
`bootstrap.py`) sits outside it. Record schemas: `contracts/internal/index-entry.schema.json`
(the point payload), `index-set.schema.json` and `index.schema.json`. The Search Index and
the model are extension points: another vector database is a module implementing
`SearchIndexReader` and `SearchIndexWriter` composed in `bootstrap.connect_search_index`,
another gateway a module implementing `EmbeddingModel` composed in
`bootstrap.create_embedding_model`; the service, scheduler and contracts do not change.

Deferred, as in the earlier steps: liveness checks and the purge of retired assets' points
(`query-api` will filter on `asset_version` meanwhile), EmbeddingGemma's shorter vector
sizes, and hybrid (lexical) postings.

## Develop

```sh
pdm run lint              # ruff check + ruff format --check
pdm run format            # ruff format
pdm run typecheck         # mypy --strict
pdm run test              # pytest with coverage (threshold 95%); MongoDB and Qdrant tests skip when none is reachable
pdm run test-integration  # only the MongoDB and Qdrant cases (stub scene-api and Ollama servers are started in-process), without the coverage threshold
```

The integration cases need a MongoDB at `SCENE_RAG_RETRIEVAL_CHUNK_STORE_URI` and, for those
marked `qdrant`, a Qdrant at `SCENE_RAG_RETRIEVAL_SEARCH_INDEX_URL` (`docker compose up -d
mongo qdrant` at the repository root serves both on localhost).

Tests are named `given_<state>_when_<event>_then_<outcome>` and grouped in one class per
subject. The fakes and builders live in `tests/support/chunking_fakes.py` (including a
test-only JSON encoder of the published documents, used to script the scene-api) and
`tests/support/embedding_fakes.py` (an in-memory Search Index and ledger, a deterministic
model, a scripted HTTP transport); `tests/conftest.py` wires them into the `chunking_harness`
and `embedding_harness` fixtures, which a test class may override under the same name to seed
state. The Search Index cases run against the fake and against Qdrant, so the fake stays
faithful. Code follows the Google Python Style Guide for
docstrings (enforced by ruff's `D` rules) and strict typing (`ANN` rules and mypy strict).
`tests/architecture/` enforces the import rules: no cross-step imports, nothing from
`scene_rag.extraction`, the environment is read only in each `bootstrap.py`, pymongo is
confined to the store modules (`chunk_store.py`; `chunk_reader.py` and `index_store.py`) and
bootstrap, HTTP to each step's `http_transport.py`, and a step's writer side is imported by
that step alone.
