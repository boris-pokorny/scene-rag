"""Step 1, `ingestion`: discover media through connectors and register asset versions.

An asset version is the content hash of the bytes a source holds; the Asset Registry keeps
every source reference of a version together with that reference's entitlements. The step
owns the registry and is its only writer.
"""
