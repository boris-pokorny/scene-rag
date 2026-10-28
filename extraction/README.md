# scene-rag-extraction

The `extraction` service of scene-rag (Media context): steps 1–3 of the indexing pipeline
and the `scene-catalog` role. Implemented so far:

- **step 1, `ingestion`**: discover media through a connector and register asset versions
  (content hash), their source references and entitlements in the Asset Registry;
- **step 2, `segmentation`**: cut every registered asset version into scenes, render keyframes
  and audio clips into the Derivative Store, and write one scene set per asset version and
  fingerprint to the Scene Store;
- **step 3, `enrichment`**: run every registered enricher (vision description over the keyframes,
  speech transcription over the audio clips) over every scene of every scene set, and write one
  record per scene, enricher and fingerprint to the Enrichment Store.

## Setup

pdm in PEP 582 mode (`pdm.toml` sets `use_venv = false`); packages land in `__pypackages__/`,
no virtualenv is created.

```sh
pdm install -G segmentation -G enrichment   # PySceneDetect and OpenCV for step 2, faster-whisper for step 3
cp .env.example .env                        # local settings; .env is ignored by git
```

Segmentation also needs `ffmpeg` on the path (or `SCENE_RAG_EXTRACTION_FFMPEG_BINARY`). The
Asset Registry, the Scene Store and the Enrichment Store are MongoDB collections. Start a server
locally:

```sh
docker run -d --name scene-rag-mongo -p 27017:27017 docker.io/mongo
```

Enrichment's vision model runs behind an Ollama server; the speech model (faster-whisper) runs
in-process and is downloaded on first use into the Hugging Face cache (or
`SCENE_RAG_EXTRACTION_WHISPER_CACHE_DIR`). Start Ollama locally and pull the model once:

```sh
docker run -d --name scene-rag-ollama -p 11434:11434 -v scene-rag-ollama:/root/.ollama docker.io/ollama/ollama
docker exec scene-rag-ollama ollama pull qwen2.5vl:3b
```

## Run

Settings (where and how fast) come from the environment; `.env` (copied from
`.env.example`, never committed) holds local values and is loaded by `pdm run`. Hyperparameters (what is produced) are the defaults of
each step's config dataclass (`IngestionConfig` in `src/scene_rag/extraction/ingestion/config.py`,
`SegmentationConfig` in `src/scene_rag/extraction/segmentation/config.py`, `EnrichmentConfig` in
`src/scene_rag/extraction/enrichment/config.py`) and change only
through a pull request, which changes the step's fingerprint.

```sh
cp some-video.mp4 data/media/
pdm run worker --step ingestion --once        # one pass; needs no poll interval
pdm run worker --step segmentation --once     # cuts what ingestion registered
pdm run worker --step enrichment --once       # describes and transcribes what segmentation cut
pdm run worker --step ingestion               # poll every SCENE_RAG_EXTRACTION_POLL_INTERVAL_SECONDS
```

Every variable not marked optional is required by the step that reads it; a missing one ends the process with a message naming it. Sample values for a local run are
in `.env.example`.

| Variable | Meaning | Read by |
|---|---|---|
| `SCENE_RAG_EXTRACTION_REGISTRY_URI` | MongoDB connection string of the Asset Registry. Credentials in it are redacted from logs. | ingestion, segmentation, enrichment |
| `SCENE_RAG_EXTRACTION_REGISTRY_DATABASE` | Database holding the registry collection. | ingestion, segmentation, enrichment |
| `SCENE_RAG_EXTRACTION_SOURCE_ROOT` | Directory the filesystem connector lists; must exist. Relative paths resolve against the working directory. | ingestion |
| `SCENE_RAG_EXTRACTION_SOURCE_ID` | Identifier stamped on every source reference the connector registers. | ingestion |
| `SCENE_RAG_EXTRACTION_SOURCE_ENTITLEMENTS` | Comma-separated entitlements granted to every file; the filesystem has no permission model of its own. At least one. | ingestion |
| `SCENE_RAG_EXTRACTION_SCENE_STORE_URI` | MongoDB connection string of the Scene Store. | segmentation, enrichment |
| `SCENE_RAG_EXTRACTION_SCENE_STORE_DATABASE` | Database holding the scene set collection. | segmentation, enrichment |
| `SCENE_RAG_EXTRACTION_DERIVATIVE_ROOT` | Directory the filesystem Derivative Store files keyframes and audio clips under; created on first use. | segmentation |
| `SCENE_RAG_EXTRACTION_FFMPEG_BINARY` | ffmpeg executable name or path; `ffmpeg` on the path by default. | segmentation (optional) |
| `SCENE_RAG_EXTRACTION_POLL_INTERVAL_SECONDS` | Seconds between two passes of the worker loop; positive. Also the first pause after a failed pass. | all; loop only, unused with `--once` |
| `SCENE_RAG_EXTRACTION_SEGMENTATION_WORKERS` | Asset versions segmented at once within a pass; `1` by default, one after another. One version already keeps about four cores busy decoding, so about the core count divided by four is worthwhile. | segmentation (optional) |
| `SCENE_RAG_EXTRACTION_AUDIO_CLIP_WORKERS` | ffmpeg processes slicing the audio clips of one version at once; `1` by default, `4` is a measured sweet spot. Up to workers × clip workers ffmpeg processes run at once. | segmentation (optional) |
| `SCENE_RAG_EXTRACTION_ENRICHMENT_STORE_URI` | MongoDB connection string of the Enrichment Store. | enrichment |
| `SCENE_RAG_EXTRACTION_ENRICHMENT_STORE_DATABASE` | Database holding the enrichment collection. | enrichment |
| `SCENE_RAG_EXTRACTION_OLLAMA_URL` | Base URL of the Ollama server answering the vision model; `http://localhost:11434` by default. | enrichment (optional) |
| `SCENE_RAG_EXTRACTION_OLLAMA_TIMEOUT_SECONDS` | Longest wait for one vision answer; `600` by default, since CPU inference takes minutes per scene. | enrichment (optional) |
| `SCENE_RAG_EXTRACTION_WHISPER_CACHE_DIR` | Directory the speech model is downloaded to and loaded from; the Hugging Face cache by default. | enrichment (optional) |
| `SCENE_RAG_EXTRACTION_WHISPER_CPU_THREADS` | Threads the speech model computes on; `0` (the library's choice) by default. | enrichment (optional) |
| `SCENE_RAG_EXTRACTION_ENRICHMENT_WORKERS` | Scene sets enriched at once within a pass; `1` by default. Ollama serialises requests unless its `OLLAMA_NUM_PARALLEL` is raised, so more workers mostly overlap the speech model with the vision model. | enrichment (optional) |

### Ingestion

A pass hashes new or changed files under the source root and issues one registry command per
change: register a version, add a source reference (same bytes under two paths: one version,
two references), refresh a reference whose metadata changed, supersede one whose file changed,
retire references of vanished files. Nothing is deleted; a version nobody holds gets a
`retired_at` tombstone.

- Files below `min_size_bytes` are quarantined: logged, counted, never registered.
- Files younger than `min_age_seconds`, or whose size or mtime moved while hashed (a copy in
  progress), are deferred to a later pass with their earlier reference kept.
- A missing source root aborts the pass rather than retiring everything.
- A failed pass in the loop is retried after a pause doubling up to five minutes.

Record schema: `contracts/internal/asset-version.schema.json`.

### Segmentation

Pending work is derived, never flagged: a pass lists the live asset versions, subtracts those
with a scene set under the current fingerprint, and segments the rest. Per version:

1. Locate a `file://` reference whose file still has the size and mtime ingestion recorded;
   otherwise the version is *deferred* until ingestion reconciles the source.
2. Cut at content changes with PySceneDetect (no cuts: one scene), render evenly spaced
   keyframes per scene, slice one 16-bit mono WAV per scene with ffmpeg (none without an audio
   stream). A missing ffmpeg ends the process at startup, not every version.
3. File keyframes and clips by content hash under
   `<derivative root>/<algorithm>/<hh>/<digest>.<ext>`, rendered first in
   `<derivative root>/.workspace`, so reruns and duplicate frames cost nothing and a failed run
   leaves nothing to undo.
4. Write the scene set last, id `hash(asset version + fingerprint)`, after a final liveness
   check (*dropped* if retired meanwhile); an insert another worker won is a *duplicate*.

A hyperparameter change gives a new fingerprint, so the next pass segments every version again
next to the old sets; nothing purges the sets of retired versions yet. Record schema:
`contracts/internal/scene-set.schema.json`.

Concurrency lives outside the fingerprint: `scheduler.py` hands pending versions to a
`Dispatcher` (serial or thread pool, picked by `bootstrap` from the worker settings), and the
audio clipper gets its own for per-scene ffmpeg processes. The fingerprinted work in
`service.py` and the backends is identical under either policy, so worker counts change how fast
a pass finishes, never what it writes.

### Enrichment

Pending work is derived, never flagged: a pass lists the live asset versions, every scene set
(whichever segmentation fingerprint cut it; old and new coexist until a release gate decides)
and, per enricher, the scenes already enriched under its fingerprint. The rest goes to a
`Dispatcher` one scene set at a time. Per enricher and pending scene:

1. The enricher reads its derivatives through a `DerivativeReader` that verifies each file
   against its content hash, then asks its model:
   - **`description`**: up to `max_keyframes` evenly spaced keyframes and the configured prompt
     go to the Ollama vision model (`qwen2.5vl:3b`, temperature 0, fixed seed); one segment
     spanning the scene.
   - **`transcript`**: the audio clip (16 kHz 16-bit PCM WAV, decoded with the standard
     library) goes to faster-whisper (`small`, int8, VAD on, no conditioning on previous
     text); its segments are offset into the stream and clamped to the scene, with language
     and a confidence derived from the average log-probability.
2. After a liveness check, one record is written: id `hash(scene set version + scene index +
   enricher + fingerprint)`, with the model digest, the classification and a lineage naming the
   scene set and every derivative read. A scene offering nothing (no keyframes, no clip, no
   speech) gets a record with no segments, so it is never retried.

Records land one scene at a time, so a killed worker keeps what it had. A scene whose derivative
is missing or altered, or whose model call fails, is logged and skipped while the rest of the
set runs; the set counts once as *deferred* or *failed*, a version retired meanwhile as
*dropped*, an insert another worker won as *duplicate*, and the next pass retries only what is
missing. Each enricher has its own fingerprint: its config section, its output-defining
modules (`models.py`, `fingerprint.py`, the enricher and its backend) and its model digest
(from Ollama's model list, or a hash of the Whisper snapshot). The HTTP transport
(`ollama_transport.py`) and the model loader (`whisper_loader.py`) sit outside it, so changing
the prompt reruns the descriptions and nothing else, and an error message or cache path
changes nothing. An unreachable Ollama, a missing model or a speech model that cannot load ends
the process at startup. Record schema: `contracts/internal/enrichment.schema.json`.

## Develop

```sh
pdm run lint          # ruff check + ruff format --check
pdm run format        # ruff format
pdm run typecheck     # mypy --strict
pdm run test              # pytest with coverage (threshold 95%); MongoDB tests skip when none is reachable
pdm run test-integration  # only the MongoDB and ffmpeg cases, without the coverage threshold
SCENE_RAG_EXTRACTION_TEST_MODELS=1 pdm run test-integration  # also the cases needing Ollama and the tiny Whisper model
```

Tests are named `given_<state>_when_<event>_then_<outcome>` and grouped in one class per
subject. The fakes and builders live in `tests/support/fakes.py`; `tests/conftest.py` wires them
into the `harness`, `fake_registry`, `segmentation_harness` and `enrichment_harness` fixtures,
which a test class may override under the same name to seed state; the segmentation and
enrichment fakes live in `tests/support/segmentation_fakes.py` and
`tests/support/enrichment_fakes.py`. Code follows the Google Python Style Guide for docstrings (enforced by ruff's `D` rules) and
strict typing (`ANN` rules and mypy strict). `tests/architecture/` enforces the import rules:
no cross-step imports (segmentation and enrichment read the registry and the scene sets through
their own views, pinned by the contracts), the environment is read only in `bootstrap.py`,
pymongo is confined to each step's store backends and bootstrap, PySceneDetect and OpenCV to
`scenedetect_backend.py`, subprocesses to `ffmpeg_backend.py`, HTTP to `ollama_transport.py`,
faster-whisper to `whisper_loader.py`, and a step's writer side is imported by that step alone.
