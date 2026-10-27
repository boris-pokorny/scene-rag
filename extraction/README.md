# scene-rag-extraction

The `extraction` service of scene-rag (Media context): steps 1–3 of the indexing pipeline
and the `scene-catalog` role. Implemented so far: **step 1, `ingestion`**: discover media
through a connector and register asset versions (content hash), their source references and
entitlements in the Asset Registry.

## Setup

pdm in PEP 582 mode (`pdm.toml` sets `use_venv = false`); packages land in `__pypackages__/`,
no virtualenv is created.

```sh
pdm install
cp .env.example .env      # local settings; .env is ignored by git
```

The Asset Registry is a MongoDB collection. Start one locally:

```sh
docker run -d --name scene-rag-mongo -p 27017:27017 docker.io/mongo
```

## Run

Settings (where and how fast) come from the environment; `.env` (copied from
`.env.example`, never committed) holds local values and is loaded by `pdm run`. Hyperparameters (what is produced) are the defaults of
`IngestionConfig` in `src/scene_rag/extraction/ingestion/config.py` and change only through
a pull request, which changes the step's fingerprint.

```sh
cp some-video.mp4 data/media/
pdm run worker --step ingestion --once     # one pass; needs no poll interval
pdm run worker --step ingestion            # poll every SCENE_RAG_EXTRACTION_POLL_INTERVAL_SECONDS
```

| Variable | Value in `.env.example` |
|---|---|
| `SCENE_RAG_EXTRACTION_REGISTRY_URI` | `mongodb://localhost:27017` |
| `SCENE_RAG_EXTRACTION_REGISTRY_DATABASE` | `scene_rag_extraction` |
| `SCENE_RAG_EXTRACTION_SOURCE_ROOT` | `data/media` |
| `SCENE_RAG_EXTRACTION_SOURCE_ID` | `local` |
| `SCENE_RAG_EXTRACTION_SOURCE_ENTITLEMENTS` | `public` |
| `SCENE_RAG_EXTRACTION_POLL_INTERVAL_SECONDS` | `30` (only the loop needs it) |

Each pass lists the source, hashes new or changed files, and issues one registry command per
change: register a version, add a source reference (the same bytes under two paths are one
version with two references), refresh a reference whose metadata (entitlements, path) changed,
supersede a reference whose file changed, retire references of files that disappeared. A pass
that fails in the loop is logged and retried after a pause that doubles with every consecutive
failure, from the poll interval up to five minutes; a source root that is missing aborts the
pass instead of retiring everything. Files failing validation (below `min_size_bytes`) are
quarantined: logged, counted, never registered. Files modified less than `min_age_seconds` ago,
or whose size or modification time moved while they were hashed (a copy still in progress), are
deferred to a later pass with their earlier reference kept. Nothing is deleted; a version nobody
holds gets a `retired_at` tombstone. The record schema is
`contracts/internal/asset-version.schema.json`.

## Develop

```sh
pdm run lint          # ruff check + ruff format --check
pdm run format        # ruff format
pdm run typecheck     # mypy --strict
pdm run test              # pytest with coverage (threshold 95%); MongoDB tests skip when none is reachable
pdm run test-integration  # only the MongoDB cases, without the coverage threshold
```

Tests are named `given_<state>_when_<event>_then_<outcome>` and grouped in one class per
subject; fakes and builders live in `tests/support/`. Code follows the Google Python Style Guide for docstrings (enforced by ruff's `D` rules) and
strict typing (`ANN` rules and mypy strict). `tests/architecture/` enforces the import rules:
no cross-step imports, the environment is read only in `bootstrap.py`, pymongo
is confined to the registry backend and bootstrap, and only `ingestion` may import the
registry's writer side.
