"""Import rules: no other service's code, environment only in bootstrap, tools confined."""

import ast
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[2] / "src" / "scene_rag" / "evaluation"

# Tools confined to named modules.
CONFINED_TOOLS = {
    "urllib.request": {"http_transport.py"},
    "http.client": {"http_transport.py"},
}


def modules_under(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)


def imported_names(module: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def attribute_accesses(module: Path) -> set[str]:
    return {
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }


ALL_MODULES = modules_under(SERVICE)


def module_id(module: Path) -> str:
    return module.relative_to(SERVICE).as_posix()


class TestGivenServiceModules:
    """Boundaries every module must respect."""

    @pytest.mark.parametrize("module", ALL_MODULES, ids=module_id)
    def test_when_imported_then_no_other_service_code(self, module: Path) -> None:
        offending = {
            name
            for name in imported_names(module)
            if name.startswith(("scene_rag.retrieval", "scene_rag.extraction"))
        }
        assert not offending, (
            f"{module_id(module)} imports {sorted(offending)}: query-api is reached over HTTP"
        )

    @pytest.mark.parametrize("module", ALL_MODULES, ids=module_id)
    def test_when_not_bootstrap_then_no_environment_access(self, module: Path) -> None:
        if module.name == "bootstrap.py":
            return
        accesses = attribute_accesses(module)
        assert "os.environ" not in accesses, f"{module_id(module)} reads os.environ"
        assert "os.getenv" not in accesses, f"{module_id(module)} reads os.getenv"

    @pytest.mark.parametrize("module", ALL_MODULES, ids=module_id)
    def test_when_not_designated_then_no_confined_tool_import(self, module: Path) -> None:
        names = imported_names(module)
        for tool, allowed in CONFINED_TOOLS.items():
            if module_id(module) in allowed:
                continue
            offending = {n for n in names if n == tool or n.startswith(tool + ".")}
            assert not offending, f"{module_id(module)} imports {tool}; only {sorted(allowed)} may"
