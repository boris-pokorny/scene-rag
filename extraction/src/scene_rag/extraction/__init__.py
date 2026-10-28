"""scene-rag `extraction` service: Media context, steps 1-3 and the scene-api role.

Each step is a subpackage with its own models, protocols, config, fingerprint and
bootstrap. Steps never import each other; roles share the service's stores.
"""
