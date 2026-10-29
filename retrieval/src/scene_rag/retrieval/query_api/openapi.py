"""Exporter of the published OpenAPI document of the query-api.

The routes of `create_app` reference ``#/components/schemas/<Name>``; this module embeds the
JSON Schemas of ``contracts/published`` under those names and writes the document next to
them. ``pdm run openapi`` regenerates it; a contract test fails when the committed copy
differs.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from scene_rag.retrieval.query_api.app import MCP_PATH, create_app
from scene_rag.retrieval.query_api.encoding import API_VERSION, TITLE
from scene_rag.retrieval.query_api.mcp_server import create_mcp_server
from scene_rag.retrieval.query_api.service import NoService

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]

OUTPUT_NAME: Final = "query-api.openapi.json"
SCHEMA_ID_PREFIX: Final = "https://scene-rag.example/retrieval/published/"
SCHEMA_FILES: Final = {
    "SearchRequest": "search-request.schema.json",
    "AnswerRequest": "answer-request.schema.json",
    "SearchResponse": "search-response.schema.json",
    "AnswerResponse": "answer-response.schema.json",
    "Source": "source.schema.json",
    "Served": "served.schema.json",
    "Error": "error.schema.json",
}
HEALTH: Final = {
    "title": "Health",
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "served"],
    "properties": {"status": {"const": "ok"}, "served": {"$ref": "#/components/schemas/Served"}},
}


def component_of(reference: str, name: str) -> str:
    """Rewrite a reference of a standalone schema to point into the OpenAPI document."""
    if reference.startswith("#/$defs/"):
        return reference.replace("#/$defs/", f"#/components/schemas/{name}/$defs/")
    if reference.startswith(SCHEMA_ID_PREFIX):
        file = reference.removeprefix(SCHEMA_ID_PREFIX)
        for other, other_file in SCHEMA_FILES.items():
            if other_file == file:
                return f"#/components/schemas/{other}"
    return reference


def _rewrite_refs(value: Any, name: str) -> Any:  # noqa: ANN401 - walks untyped JSON
    if isinstance(value, dict):
        return {
            key: (
                component_of(item, name)
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

    The document identifiers go, since the schema no longer stands alone, and its references,
    local or to a sibling schema, are rewritten to point into the OpenAPI document.
    """
    embedded = {key: value for key, value in schema.items() if key not in ("$schema", "$id")}
    rewritten: Json = _rewrite_refs(embedded, name)
    return rewritten


def load_schemas(directory: Path) -> dict[str, Json]:
    """Load the published schemas and derive the health shape."""
    schemas = {
        name: embed(name, json.loads((directory / file).read_text()))
        for name, file in SCHEMA_FILES.items()
    }
    schemas["Health"] = dict(HEALTH)
    return schemas


def build_openapi(app: FastAPI, schemas: Mapping[str, Json]) -> Json:
    """Generate the OpenAPI document of ``app`` with ``schemas`` as its components.

    The MCP transport route is left out: its protocol is documented by the MCP tool schemas.
    """
    routes = [route for route in app.routes if getattr(route, "path", None) != MCP_PATH]
    document = get_openapi(
        title=TITLE,
        version=API_VERSION,
        openapi_version="3.1.0",
        description="Search and answers over the scene-rag Search Index, for evaluation, "
        "user interfaces and services. Read-only. The caller's entitlements travel in the "
        "`X-Scene-Rag-Entitlements` header, asserted by the gateway in front.",
        routes=routes,
    )
    for path in document["paths"].values():
        for operation in path.values():
            # FastAPI documents its own 422; the application answers 400 instead (`app.py`).
            operation["responses"].pop("422", None)
            operation.pop("requestBody", None)
    for name, request in (("/search", "SearchRequest"), ("/answer", "AnswerRequest")):
        document["paths"][name]["post"]["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {"schema": {"$ref": f"#/components/schemas/{request}"}}
            },
        }
    document.setdefault("components", {})["schemas"] = dict(schemas)
    return document


def render(document: Json) -> str:
    """Render the document the way it is committed: sorted keys, two-space indent."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def generate(contracts: Path) -> str:
    """Render the document for the schemas in ``contracts``."""
    service = NoService()
    return render(
        build_openapi(
            create_app(service, create_mcp_server(service), allowed_hosts=("*",)),
            load_schemas(contracts),
        )
    )


def default_contracts_dir() -> Path:
    """Return ``contracts/published`` of the service checkout this module is installed from."""
    return Path(__file__).resolve().parents[4] / "contracts" / "published"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the document; ``--contracts`` names the directory, the checkout's by default."""
    parser = argparse.ArgumentParser(prog="query-api openapi")
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
