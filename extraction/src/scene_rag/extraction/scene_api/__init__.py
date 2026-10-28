"""The `scene-api` role of the extraction service: scene records over HTTP.

A read model over the stores the steps write (the Asset Registry, the Scene Store, the
Enrichment Store and the Derivative Store), joined into scene records for the `retrieval`
service. The role writes nothing, has no hyperparameters and no fingerprint: a change to it
deploys directly. It imports no step; it reads their collections through its own views, pinned
by ``contracts/internal``, and publishes ``contracts/published``.
"""
