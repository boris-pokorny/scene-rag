"""scene-rag `evaluation` service: step 7, offline evaluation of what query-api serves.

Two commands: `run` puts a versioned dataset to query-api over HTTP and stores an eval run in
the Eval Store; `compare` reads two stored runs back and writes the promotion gate's verdict.
Runs are stored rather than compared in one go because the baseline usually cannot be run
again: once the alias moves, query-api serves only the candidate. Retrieval data reaches this
service only through query-api, so a run measures exactly what is served.
"""
