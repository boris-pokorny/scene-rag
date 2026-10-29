"""Exporter of the published MCP document of the query-api: tools and resource templates.

What a client learns from ``tools/list`` and ``resources/templates/list``, with the input and
output schemas the SDK derives from the adapter's signatures, written next to the OpenAPI
document. ``pdm run mcp-schemas`` regenerates it; a contract test fails when the committed
copy differs.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import anyio
from mcp.server.mcpserver import MCPServer

from scene_rag.retrieval.query_api.mcp_server import create_mcp_server
from scene_rag.retrieval.query_api.service import NoService

# JSON documents are untyped; `Any` is confined to this boundary.
type Json = dict[str, Any]

OUTPUT_NAME: Final = "query-api.mcp.json"


async def describe(server: MCPServer) -> Json:
    """Return what the server advertises: identity, tools, resource templates."""
    tools = await server.list_tools()
    templates = await server.list_resource_templates()
    return {
        "server": {
            "name": server.name,
            "version": server.version,
            "instructions": server.instructions,
        },
        "tools": [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools],
        "resource_templates": [
            template.model_dump(mode="json", by_alias=True, exclude_none=True)
            for template in templates
        ],
    }


def render(document: Json) -> str:
    """Render the document the way it is committed: sorted keys, two-space indent."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def generate() -> str:
    """Render the document of the server as `create_mcp_server` builds it."""
    return render(anyio.run(describe, create_mcp_server(NoService())))


def default_contracts_dir() -> Path:
    """Return ``contracts/published`` of the service checkout this module is installed from."""
    return Path(__file__).resolve().parents[4] / "contracts" / "published"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the document; ``--contracts`` names the directory, the checkout's by default."""
    parser = argparse.ArgumentParser(prog="query-api mcp-schemas")
    parser.add_argument(
        "--contracts",
        type=Path,
        default=default_contracts_dir(),
        help="directory the document is written to",
    )
    args = parser.parse_args(argv)
    output = args.contracts / OUTPUT_NAME
    output.write_text(generate())
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
