# scene-rag-extraction

The `extraction` service of scene-rag (Media context): steps 1–3 of the indexing pipeline
and the `scene-catalog` role. Implemented so far:

- **step 1, `ingestion`**: discover media through a connector and register asset versions
  (content hash), their source references and entitlements in the Asset Registry;
- **step 2, `segmentation`**: cut every registered asset version into scenes, render keyframes
  and audio clips into the Derivative Store, and write one scene set per asset version and
  fingerprint to the Scene Store.

## Setup

pdm in PEP 582 mode (`pdm.toml` sets `use_venv = false`); packages land in `__pypackages__/`,
no virtualenv is created.

```sh
pdm install -G segmentation   # the extra brings PySceneDetect and OpenCV for step 2
cp .env.example .env          # local settings; .env is ignored by git
```

Segmentation also needs `ffmpeg` on the path (or `SCENE_RAG_EXTRACTION_FFMPEG_BINARY`). The
Asset Registry and the Scene Store are MongoDB collections. Start a server locally:

```sh
docker run -d --name scene-rag-mongo -p 27017:27017 docker.io/mongo
```

## Run

Settings (where and how fast) come from the environment; `.env` (copied from
`.env.example`, never committed) holds local values and is loaded by `pdm run`. Hyperparameters (what is produced) are the defaults of
each step's config dataclass (`IngestionConfig` in `src/scene_rag/extraction/ingestion/config.py`,
`SegmentationConfig` in `src/scene_rag/extraction/segmentation/config.py`) and change only
through a pull request, which changes the step's fingerprint.

```sh
cp some-video.mp4 data/media/
pdm run worker --step ingestion --once        # one pass; needs no poll interval
pdm run worker --step segmentation --once     # cuts what ingestion registered
pdm run worker --step ingestion               # poll every SCENE_RAG_EXTRACTION_POLL_INTERVAL_SECONDS
```

Every variable but the poll interval and the ffmpeg binary is required by the step that reads
it; a missing one ends the process with a message naming it. Sample values for a local run are
in `.env.example`.

| Variable | Meaning | Read by |
|---|---|---|
| `SCENE_RAG_EXTRACTION_REGISTRY_URI` | MongoDB connection string of the Asset Registry. Credentials in it are redacted from logs. | ingestion, segmentation |
| `SCENE_RAG_EXTRACTION_REGISTRY_DATABASE` | Database holding the registry collection. | ingestion, segmentation |
| `SCENE_RAG_EXTRACTION_SOURCE_ROOT` | Directory the filesystem connector lists; must exist. Relative paths resolve against the working directory. | ingestion |
| `SCENE_RAG_EXTRACTION_SOURCE_ID` | Identifier stamped on every source reference the connector registers. | ingestion |
| `SCENE_RAG_EXTRACTION_SOURCE_ENTITLEMENTS` | Comma-separated entitlements granted to every file; the filesystem has no permission model of its own. At least one. | ingestion |
| `SCENE_RAG_EXTRACTION_SCENE_STORE_URI` | MongoDB connection string of the Scene Store. | segmentation |
| `SCENE_RAG_EXTRACTION_SCENE_STORE_DATABASE` | Database holding the scene set collection. | segmentation |
| `SCENE_RAG_EXTRACTION_DERIVATIVE_ROOT` | Directory the filesystem Derivative Store files keyframes and audio clips under; created on first use. | segmentation |
| `SCENE_RAG_EXTRACTION_FFMPEG_BINARY` | ffmpeg executable name or path; `ffmpeg` on the path by default. | segmentation (optional) |
| `SCENE_RAG_EXTRACTION_POLL_INTERVAL_SECONDS` | Seconds between two passes of the worker loop; positive. Also the first pause after a failed pass. | both; loop only, unused with `--once` |

### Ingestion

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

### Segmentation

Pending work is derived, never flagged: a pass lists the live asset versions and the asset
versions that already have a scene set under the step's current fingerprint, and processes
the difference. For each version it finds a `file://` source reference whose file still has
the size and modification time ingestion recorded (otherwise the version is *deferred* until
ingestion has reconciled the source), cuts the video at content changes with PySceneDetect
(a video without cuts is one scene), renders evenly spaced keyframes per scene and slices one
16-bit mono WAV clip per scene with ffmpeg (none when the asset has no audio stream). Keyframes
and clips are rendered into a scratch directory under `<derivative root>/.workspace` and filed
by content hash under `<derivative root>/<algorithm>/<hh>/<digest>.<ext>`, so reruns and
duplicate frames cost nothing and a failed run leaves nothing to undo. A missing ffmpeg ends
the process at startup rather than failing every version. The scene
set is written last, under `hash(asset version + fingerprint)` as its id, after a final check
that the version is still live (*dropped* otherwise); an insert refused because another worker
wrote it first is a *duplicate*. A hyperparameter change gives a new fingerprint, so the next
pass segments every version again next to the old sets. The record schema is
`contracts/internal/scene-set.schema.json`. Nothing purges the sets of retired versions yet.

## Develop

```sh
pdm run lint          # ruff check + ruff format --check
pdm run format        # ruff format
pdm run typecheck     # mypy --strict
pdm run test              # pytest with coverage (threshold 95%); MongoDB tests skip when none is reachable
pdm run test-integration  # only the MongoDB and ffmpeg cases, without the coverage threshold
```

Tests are named `given_<state>_when_<event>_then_<outcome>` and grouped in one class per
subject. The fakes and builders live in `tests/support/fakes.py`; `tests/conftest.py` wires them
into the `harness`, `fake_registry` and `segmentation_harness` fixtures, which a test class may
override under the same name to seed state; the segmentation fakes live in
`tests/support/segmentation_fakes.py`. Code follows the Google Python Style Guide for docstrings (enforced by ruff's `D` rules) and
strict typing (`ANN` rules and mypy strict). `tests/architecture/` enforces the import rules:
no cross-step imports (segmentation reads the registry through its own view, pinned by the
asset version contract), the environment is read only in `bootstrap.py`, pymongo is confined to
each step's store backends and bootstrap, PySceneDetect and OpenCV to `scenedetect_backend.py`,
subprocesses to `ffmpeg_backend.py`, and a step's writer side is imported by that step alone.
