"""scene-rag `retrieval` service: Retrieval context, steps 4-5 and the query-api role.

Each step is a subpackage with its own models, protocols, config, fingerprint and
bootstrap. Steps never import each other; roles share the service's stores. Media data
reaches this service only through the `scene-api` of `extraction`.
"""
