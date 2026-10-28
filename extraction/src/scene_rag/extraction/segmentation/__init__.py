"""Step 2, `segmentation`: cut asset versions into scenes, keyframes and audio clips.

The step reads live asset versions from the Asset Registry (its own read model over the
records ingestion writes), writes one scene set per asset version and fingerprint to the Scene
Store, and files keyframes and audio clips content-addressed in the Derivative Store. It owns
both stores and is their only writer.
"""
