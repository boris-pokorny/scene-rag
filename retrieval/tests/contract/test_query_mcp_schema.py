"""The published MCP document: committed as generated, and true to the published samples."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scene_rag.retrieval.query_api import mcp_schema
from scene_rag.retrieval.query_api.mcp_server import KEYFRAME_URI, SCENE_URI

PUBLISHED = Path(__file__).resolve().parents[2] / "contracts" / "published"
DOCUMENT = json.loads((PUBLISHED / mcp_schema.OUTPUT_NAME).read_text())
TOOLS = {tool["name"]: tool for tool in DOCUMENT["tools"]}


def sample(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((PUBLISHED / "samples" / f"{name}.json").read_text())
    return document


class TestGivenCommittedDocument:
    """The file under contracts/published."""

    def test_when_regenerated_then_identical(self) -> None:
        assert mcp_schema.generate() == (PUBLISHED / mcp_schema.OUTPUT_NAME).read_text()

    def test_when_inspected_then_two_tools_two_templates(self) -> None:
        assert list(TOOLS) == ["search_scenes", "answer"]
        assert [t["uriTemplate"] for t in DOCUMENT["resource_templates"]] == [
            SCENE_URI,
            KEYFRAME_URI,
        ]
        assert DOCUMENT["server"]["name"] == "scene-rag-query-api"
        for tool in TOOLS.values():
            assert tool["annotations"] == {"readOnlyHint": True}
            assert tool["inputSchema"]["required"] == ["question"]

    @pytest.mark.parametrize(
        "tool, name", [("search_scenes", "search-response"), ("answer", "answer-response")]
    )
    def test_when_sample_validated_against_output_schema_then_accepted(
        self, tool: str, name: str
    ) -> None:
        schema = TOOLS[tool]["outputSchema"]
        jsonschema.Draft202012Validator.check_schema(schema)

        jsonschema.Draft202012Validator(schema).validate(sample(name))

    def test_when_written_by_main_then_file_matches(self, tmp_path: Path) -> None:
        assert mcp_schema.main(["--contracts", str(tmp_path)]) == 0
        assert (tmp_path / mcp_schema.OUTPUT_NAME).read_text() == mcp_schema.generate()
