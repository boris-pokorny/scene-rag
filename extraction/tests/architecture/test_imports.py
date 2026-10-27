"""Import rules of the service: step isolation, settings bound only in bootstrap."""

import ast
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[2] / "src" / "scene_rag" / "extraction"
INGESTION = SERVICE / "ingestion"
STEPS = {"ingestion", "segmentation", "enrichment"}
ROLES = {"catalog"}


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


OUTSIDE_INGESTION = [m for m in modules_under(SERVICE) if INGESTION not in m.parents]


class TestGivenServiceModules:
    """Boundaries every module of the service must respect."""

    @pytest.mark.parametrize("module", modules_under(INGESTION), ids=lambda p: p.name)
    def test_when_in_ingestion_then_no_sibling_step_import(self, module: Path) -> None:
        foreign = {f"scene_rag.extraction.{package}" for package in (STEPS | ROLES) - {"ingestion"}}
        offending = {
            name for name in imported_names(module) if any(name.startswith(f) for f in foreign)
        }
        assert not offending, f"{module.name} imports {sorted(offending)}"

    @pytest.mark.parametrize("module", modules_under(INGESTION), ids=lambda p: p.name)
    def test_when_not_bootstrap_then_no_environment_access(self, module: Path) -> None:
        if module.name == "bootstrap.py":
            return
        assert "os.environ" not in attribute_accesses(module), f"{module.name} reads os.environ"
        assert "os.getenv" not in attribute_accesses(module), f"{module.name} reads os.getenv"

    @pytest.mark.parametrize("module", modules_under(INGESTION), ids=lambda p: p.name)
    def test_when_not_registry_then_no_pymongo_import(self, module: Path) -> None:
        if module.name in {"registry.py", "bootstrap.py"}:
            return
        uses_pymongo = any(
            name.split(".")[0] in {"pymongo", "bson"} for name in imported_names(module)
        )
        assert not uses_pymongo, f"{module.name} imports pymongo"

    @pytest.mark.parametrize("module", OUTSIDE_INGESTION, ids=lambda p: p.name)
    def test_when_outside_ingestion_then_no_writer_import(self, module: Path) -> None:
        names = imported_names(module)
        assert "scene_rag.extraction.ingestion.protocols.AssetRegistryWriter" not in names
        assert "scene_rag.extraction.ingestion.registry" not in names, (
            f"{module.name} must go through bootstrap, not the registry backend"
        )
