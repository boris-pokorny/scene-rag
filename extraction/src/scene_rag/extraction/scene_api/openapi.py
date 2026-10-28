"""Exporter of the published OpenAPI document of the scene-api.

The routes of `create_app` reference ``#/components/schemas/<Name>``; this module embeds the
JSON Schemas of ``contracts/published`` under those names and writes the document next to
them. ``pdm run openapi`` regenerates it; a contract test fails when the committed copy
differs.
"""

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from scene_rag.extraction.scene_api import models
from scene_rag.extraction.scene_api.app import API_VERSION, TITLE, create_app
from scene_rag.extraction.scene_api.protocols import SceneCatalog

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]

OUTPUT_NAME: Final = "scene-api.openapi.json"
SCHEMA_FILES: Final = {
    "SceneRecord": "scene-record.schema.json",
    "SceneSetSummary": "scene-set-summary.schema.json",
    "AssetVersion": "asset-version.schema.json",
    "Page": "page.schema.json",
    "Error": "error.schema.json",
}
PAGES: Final = {"PageOfSceneSetSummary": "SceneSetSummary", "PageOfSceneRecord": "SceneRecord"}
HEALTH: Final = {
    "title": "Health",
    "type": "object",
    "additionalProperties": False,
    "required": ["status"],
    "properties": {"status": {"const": "ok"}},
}


def _rewrite_refs(value: Any, name: str) -> Any:  # noqa: ANN401 - walks untyped JSON
    if isinstance(value, dict):
        return {
            key: (
                item.replace("#/$defs/", f"#/components/schemas/{name}/$defs/")
                if key == "$ref" and isinstance(item, str)
                else _rewrite_refs(item, name)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_refs(item, name) for item in value]
    return value


def embed(name: str, schema: Json) -> Json:
    """Prepare a standalone JSON Schema for ``components.schemas[name]``.

    The document identifiers go, since the schema no longer stands alone, and its local
    references are rewritten to point into the OpenAPI document.
    """
    embedded = {key: value for key, value in schema.items() if key not in ("$schema", "$id")}
    rewritten: Json = _rewrite_refs(embedded, name)
    return rewritten


def load_schemas(directory: Path) -> dict[str, Json]:
    """Load the published schemas and derive the page variants and the health shape."""
    schemas = {
        name: embed(name, json.loads((directory / file).read_text()))
        for name, file in SCHEMA_FILES.items()
    }
    for page_name, item_name in PAGES.items():
        page = copy.deepcopy(schemas["Page"])
        page["title"] = page_name
        page["properties"]["items"]["items"] = {"$ref": f"#/components/schemas/{item_name}"}
        schemas[page_name] = page
    schemas["Health"] = dict(HEALTH)
    return schemas


def build_openapi(app: FastAPI, schemas: Mapping[str, Json]) -> Json:
    """Generate the OpenAPI document of ``app`` with ``schemas`` as its components."""
    document = get_openapi(
        title=TITLE,
        version=API_VERSION,
        openapi_version="3.1.0",
        description="Scene records of the scene-rag extraction service, for the retrieval "
        "service. Read-only.",
        routes=app.routes,
    )
    for path in document["paths"].values():
        for operation in path.values():
            # FastAPI documents its own 422; the application answers 400 instead (`app.py`).
            operation["responses"].pop("422", None)
    document.setdefault("components", {})["schemas"] = dict(schemas)
    return document


def render(document: Json) -> str:
    """Render the document the way it is committed: sorted keys, two-space indent."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


class _NoCatalog:
    """`SceneCatalog` that is never called: the routes are introspected, not run."""

    def list_scene_sets(self, query: models.SceneSetQuery) -> models.Page[models.SceneSetSummary]:
        raise NotImplementedError

    def get_scene_set(self, version: str) -> models.SceneSetSummary:
        raise NotImplementedError

    def list_scene_records(
        self, version: str, query: models.SceneQuery
    ) -> models.Page[models.SceneRecord]:
        raise NotImplementedError

    def get_scene_record(
        self, version: str, index: int, pins: Sequence[models.EnrichmentPin]
    ) -> models.SceneRecord:
        raise NotImplementedError

    def get_asset_version(self, version: str) -> models.AssetVersionView:
        raise NotImplementedError

    def find_derivative(self, digest: str) -> models.DerivativeRefView:
        raise NotImplementedError

    def get_derivative(self, digest: str) -> models.DerivativeContent:
        raise NotImplementedError


def generate(contracts: Path) -> str:
    """Render the document for the schemas in ``contracts``."""
    catalog: SceneCatalog = _NoCatalog()
    return render(build_openapi(create_app(catalog), load_schemas(contracts)))


def default_contracts_dir() -> Path:
    """Return ``contracts/published`` of the service checkout this module is installed from."""
    return Path(__file__).resolve().parents[4] / "contracts" / "published"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the document; ``--contracts`` names the directory, the checkout's by default."""
    parser = argparse.ArgumentParser(prog="scene-api openapi")
    parser.add_argument(
        "--contracts",
        type=Path,
        default=default_contracts_dir(),
        help="directory holding the published schemas; the document is written next to them",
    )
    args = parser.parse_args(argv)
    output = args.contracts / OUTPUT_NAME
    output.write_text(generate(args.contracts))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
