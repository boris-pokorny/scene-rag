# scene-rag-evaluation

The `evaluation` service of scene-rag, step 7: offline evaluation of what the retrieval
service's `query-api` serves. Two commands:

- **`run`**: put every case of a versioned dataset to query-api over HTTP (`POST /search`
  and `POST /answer`, under the case's entitlements), score it, roll the scores up overall
  and per tag, and store the eval run in the Eval Store (`runs/<run_id>.json`).
- **`compare`**: diff a candidate run against a baseline run and decide the promotion gate;
  print the differences and write the verdict (`verdicts/<run_id>.json`), the contract a
  release step reads.

Retrieval data reaches this service only through query-api, so a run measures exactly what
is served. Each run records the dataset (name, version, file digest), what query-api reported
serving (`query_config`, index fingerprints, collection), the run's own hyperparameters, the
per-case scores and the roll-ups. If query-api starts serving something else during a run,
the run fails and nothing is stored.

## Setup

pdm in PEP 582 mode (`use_venv = false`): packages land in `__pypackages__/`. The service has
no runtime dependencies; `pdm install` brings only the dev tools.

```sh
pdm install
cp .env.example .env          # local settings, loaded by `pdm run`; ignored by git
```

It needs a running query-api (`pdm run query-api` in `../retrieval`, or Docker Compose).

## Run

```sh
pdm run run-eval --dataset data/example/1.0.0.jsonl
pdm run compare --candidate run-<id> --baseline run-<id>   # exit 0 passed, 3 failed
pdm run compare --candidate run-<id>                       # first release: absolute rules only
```

All variables are prefixed `SCENE_RAG_EVALUATION_`; sample values are in `.env.example`.

| Variable | Meaning (default) | Read by |
|---|---|---|
| `QUERY_API_URL` | Base URL of retrieval's query-api; probed at startup. | run |
| `QUERY_API_TIMEOUT_SECONDS` | Longest wait for one answer (`300`). | run (optional) |
| `EVAL_STORE_ROOT` | Directory of the Eval Store (`runs/`, `verdicts/`). | run, compare |

Hyperparameters are the defaults of `RunConfig` (cut-offs `ks`, the search `top_k`, the use
cases) and `GateConfig` (primary metrics, tolerance, zero metrics). They belong to no release
and are recorded in every run and verdict.

### Run with Docker Compose

The `evaluation` service in `../compose.yaml` sits under the `evaluation` profile, so a plain
`up` leaves it out. It mounts `data/` read-only at `/datasets` and keeps the Eval Store in a volume:

```sh
cd ..
docker compose up -d
docker compose run --rm evaluation run --dataset /data/example/1.0.0.jsonl
docker compose run --rm evaluation compare --candidate run-<id> --baseline run-<id>
```

## Datasets

Versioned JSONL in git, one case per line, at `data/<name>/<major.minor.patch>.jsonl`, next to the local Eval Store
(`data/eval-store/`, ignored by git), as extraction keeps `data/media/` next to its outputs.
Bump the major when label meaning changes; any edit needs a new version, because `compare`
refuses runs whose dataset digests differ. Relevance is labelled by **asset version and time
span**, not by chunk, so one dataset stays valid across chunking strategies and rebuilds:

```json
{"id": "q-017",
 "question": "When does the presenter switch to the architecture diagram?",
 "relevant": [{"asset_id": "asset-3fa…", "asset_version": "sha256:8c2…",
               "start_seconds": 312.0, "end_seconds": 341.5}],
 "forbidden": [],
 "reference_answer": "Around 5:12, right after the demo.",
 "tags": ["visual", "temporal"],
 "caller_entitlements": ["team-platform"]}
```

- `"relevant": []` marks an unanswerable question: the correct answer is a refusal.
- `forbidden` lists spans the caller must never see (access-control cases); a returned source
  that overlaps one is a leak. query-api does not publish entitlements, so leaks are labelled
  here.
- Asset versions come from extraction's scene-api (`GET /scene-sets` lists them) or from the
  `asset_version` of any query-api source.

The whole file is validated before the first case is asked; the shape is
`contracts/internal/dataset-case.schema.json`. `data/example/1.0.0.jsonl` uses the asset
version of retrieval's contract samples: it shows the format and drives the tests, but
matches nothing in a real index. Label your own media for a meaningful run.

## Metrics

A source is relevant when it overlaps a labelled span of the same asset version (spans that
only touch at an endpoint do not). Scores are keyed `<use case>/<metric>`.

| Metric | Use case | Rule | Roll-up |
|---|---|---|---|
| `hit@k`, `recall@k`, `ndcg@k`, `mrr` | search | span overlap; answerable cases only; a span is credited once | mean |
| `leakage` | search, answer | sources overlapping a `forbidden` span | sum (must be 0) |
| `citation_validity` | answer | no invalid citation; refusals skipped | mean |
| `refusal_correctness` | answer | refused exactly when unanswerable | mean |
| `latency_ms` | search, answer | the runner's clock | p50, p95 |
| `prompt_tokens`, `completion_tokens` | answer | as the model reports them | sum |

Searches ask for `top_k` sources (10, so `@10` can be measured); answers are asked without
a count, so the generator sees exactly what query-api serves by default. A metric that
applied to no case is left out of the roll-ups rather than reported as zero.

Faithfulness and correctness need a `Judge`, a model other than the generator. Its Protocol
is in `protocols.py`; no implementation ships yet. A judge will be wrapped in `Metric`s,
so the runner will not change when one is added.

A failed query-api call is recorded on its case and the run carries on; the gate counts the
errors.

## Promotion gate

`compare` passes a candidate only when:

1. it had no failed query-api call;
2. every zero metric (`search/leakage`, `answer/leakage`) is zero;
3. no primary metric (`search/recall@5`) is more than the tolerance (`0.02`) below the
   baseline's, and the candidate reports it.

Runs over different datasets (name, version or digest) are refused rather than judged. With
no baseline (a first release) only rules 1 and 2 and the presence of the primary metrics
apply.

## Develop

```sh
pdm run lint        # ruff check + format check
pdm run format
pdm run typecheck   # mypy --strict
pdm run test        # pytest with coverage (fail under 95 %)
```

One flat package serves both commands: `run` and `compare` are one-shot processes of one
image, released together, so they share models and one composition root (`bootstrap.py`).
What they hand over is the stored run, written and read back by `eval_store.py`; its decoder
checks `schema_version`, since a baseline run may be older than the code comparing it. The
format is pinned by `contracts/internal/eval-run.schema.json` and a write/read round-trip test.
The decoder of query-api's responses (`query_api_client.py`) is pinned by copies of
retrieval's published samples in `tests/contract/samples/`, refreshed by hand when retrieval
publishes a new major of the query API.
