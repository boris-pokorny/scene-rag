# scene-rag-retrieval

The `retrieval` service of scene-rag (Retrieval context): steps 4–5 of the indexing pipeline
and the `query-api` role. Implemented so far:

- **step 4, `chunking`**: read scene records through the `scene-api` of the `extraction`
  service, cut every scene set that is complete under the accepted upstream fingerprints
  into chunks with the registered `Chunker` (per scene by default), and write the chunks and
  one chunk set per scene set and fingerprint to the Chunk Store.

The service reads Media data only through `scene-api`: it never opens an extraction store.

## Setup

pdm in PEP 582 mode (`pdm.toml` sets `use_venv = false`); packages land in `__pypackages__/`,
no virtualenv is created.

```sh
pdm install
cp .env.example .env                        # local settings; .env is ignored by git
```

The Chunk Store is two MongoDB collections. Start a server locally, and a `scene-api` over
filled stores (`pdm run scene-api` in `../extraction`, or [Run with Docker
Compose](#run-with-docker-compose) for everything at once):

```sh
docker run -d --name scene-rag-mongo -p 27017:27017 docker.io/mongo
```

## Run

Settings (where and how fast) come from the environment; `.env` (copied from
`.env.example`, never committed) holds local values and is loaded by `pdm run`.
Hyperparameters (what is produced) are the defaults of the step's config dataclass
(`ChunkingConfig` in `src/scene_rag/retrieval/chunking/config.py`) and change only through a
pull request, which changes the step's fingerprint.

```sh
pdm run worker --step chunking --once         # one pass; needs no poll interval
pdm run worker --step chunking                # poll every SCENE_RAG_RETRIEVAL_POLL_INTERVAL_SECONDS
```

Every variable not marked optional is required; a missing one ends the process with a message
naming it. Sample values for a local run are in `.env.example`.

| Variable | Meaning | Read by |
|---|---|---|
| `SCENE_RAG_RETRIEVAL_SCENE_API_URL` | Base URL of the extraction service's scene-api. Probed at startup; an unreachable API ends the process. | chunking |
| `SCENE_RAG_RETRIEVAL_SCENE_API_TIMEOUT_SECONDS` | Longest wait for one scene-api answer; `30` by default. | chunking (optional) |
| `SCENE_RAG_RETRIEVAL_CHUNK_STORE_URI` | MongoDB connection string of the Chunk Store. Credentials in it are redacted from logs. | chunking |
| `SCENE_RAG_RETRIEVAL_CHUNK_STORE_DATABASE` | Database holding the `chunks` and `chunk_sets` collections. | chunking |
| `SCENE_RAG_RETRIEVAL_POLL_INTERVAL_SECONDS` | Seconds between two passes of the worker loop; positive. Also the first pause after a failed pass. | loop only, unused with `--once` |
| `SCENE_RAG_RETRIEVAL_CHUNKING_WORKERS` | Scene sets chunked at once within a pass; `1` by default. The work is HTTP and store round trips, so a few workers overlap them. | chunking (optional) |

### Run with Docker Compose

`compose.yaml` at the repository root wires a complete instance: the extraction service
(MongoDB, Ollama, the three workers, `scene-api`) and this service's `chunking` worker, built
from `Dockerfile` here, reading `http://scene-api:8080` and writing the `scene_rag_retrieval`
database on the same MongoDB. `CHUNKING_WORKERS` (1) and `POLL_INTERVAL_SECONDS` (30) can be
overridden from the shell or a `.env` next to `compose.yaml`.

```sh
cd ..                                         # the repository root
docker compose up --build -d
docker compose logs -f chunking
docker compose run --rm chunking worker --step chunking --once   # one pass instead of the loop
mongosh scene_rag_retrieval --eval 'db.chunk_sets.countDocuments(); db.chunks.findOne()'
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

## Develop

```sh
pdm run lint              # ruff check + ruff format --check
pdm run format            # ruff format
pdm run typecheck         # mypy --strict
pdm run test              # pytest with coverage (threshold 95%); MongoDB tests skip when none is reachable
pdm run test-integration  # only the MongoDB cases (a stub scene-api is started in-process), without the coverage threshold
```

Tests are named `given_<state>_when_<event>_then_<outcome>` and grouped in one class per
subject. The fakes and builders live in `tests/support/chunking_fakes.py` (including a
test-only JSON encoder of the published documents, used to script the scene-api);
`tests/conftest.py` wires them into the `chunking_harness` fixture, which a test class may
override under the same name to seed state. Code follows the Google Python Style Guide for
docstrings (enforced by ruff's `D` rules) and strict typing (`ANN` rules and mypy strict).
`tests/architecture/` enforces the import rules: no cross-step imports, nothing from
`scene_rag.extraction`, the environment is read only in `bootstrap.py`, pymongo is confined to
`chunk_store.py` and bootstrap, HTTP to `http_transport.py`, and the step's writer side is
imported by the step alone.
