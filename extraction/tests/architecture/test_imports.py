"""Import rules: step and role isolation, environment only in bootstrap, tools confined."""

import ast
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[2] / "src" / "scene_rag" / "extraction"
STEPS = {"ingestion", "segmentation", "enrichment"}
ROLES = {"scene_api"}
# Steps and roles alike: a subpackage with its own models, protocols, settings and bootstrap.
PACKAGES = STEPS | ROLES

# Per package: the modules that may import pymongo, besides bootstrap.
STORE_BACKENDS = {
    "ingestion": {"registry.py"},
    "segmentation": {"asset_registry.py", "scene_store.py"},
    "enrichment": {"asset_registry.py", "scene_store.py", "enrichment_store.py"},
    "scene_api": {"asset_registry.py", "scene_store.py", "enrichment_store.py"},
}
# Per step: the write side other steps and roles must never import.
WRITER_SIDE = {
    "ingestion": {
        "scene_rag.extraction.ingestion.protocols.AssetRegistryWriter",
        "scene_rag.extraction.ingestion.registry",
    },
    "segmentation": {
        "scene_rag.extraction.segmentation.protocols.SceneStoreWriter",
        "scene_rag.extraction.segmentation.scene_store",
        "scene_rag.extraction.segmentation.derivative_store",
    },
    "enrichment": {
        "scene_rag.extraction.enrichment.protocols.EnrichmentStoreWriter",
        "scene_rag.extraction.enrichment.enrichment_store",
    },
}
# Tools confined to named modules: a package, or a module within one, per tool.
CONFINED_TOOLS = {
    "scenedetect": {"segmentation/scenedetect_backend.py"},
    "cv2": {"segmentation/scenedetect_backend.py"},
    "subprocess": {"segmentation/ffmpeg_backend.py"},
    "faster_whisper": {"enrichment/whisper_loader.py"},
    "urllib.request": {"enrichment/ollama_transport.py"},
    "fastapi": {"scene_api/app.py", "scene_api/openapi.py", "scene_api/bootstrap.py"},
    "starlette": {"scene_api/app.py"},
    "uvicorn": {"scene_api/bootstrap.py"},
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


def package_of(module: Path) -> str | None:
    relative = module.relative_to(SERVICE).parts
    return relative[0] if len(relative) > 1 and relative[0] in PACKAGES else None


PACKAGE_MODULES = [module for module in modules_under(SERVICE) if package_of(module) is not None]
ROLE_MODULES = [module for module in PACKAGE_MODULES if package_of(module) in ROLES]
ALL_MODULES = modules_under(SERVICE)


def module_id(module: Path) -> str:
    return module.relative_to(SERVICE).as_posix()


class TestGivenPackageModules:
    """Boundaries every module inside a step or role must respect."""

    @pytest.mark.parametrize("module", PACKAGE_MODULES, ids=module_id)
    def test_when_in_package_then_no_sibling_import(self, module: Path) -> None:
        package = package_of(module)
        foreign = {f"scene_rag.extraction.{other}" for other in PACKAGES - {package}}
        offending = {
            name for name in imported_names(module) if any(name.startswith(f) for f in foreign)
        }
        assert not offending, f"{module_id(module)} imports {sorted(offending)}"

    @pytest.mark.parametrize("module", PACKAGE_MODULES, ids=module_id)
    def test_when_not_bootstrap_then_no_environment_access(self, module: Path) -> None:
        if module.name == "bootstrap.py":
            return
        accesses = attribute_accesses(module)
        assert "os.environ" not in accesses, f"{module_id(module)} reads os.environ"
        assert "os.getenv" not in accesses, f"{module_id(module)} reads os.getenv"

    @pytest.mark.parametrize("module", PACKAGE_MODULES, ids=module_id)
    def test_when_not_store_backend_then_no_pymongo_import(self, module: Path) -> None:
        package = package_of(module)
        assert package is not None
        if module.name in STORE_BACKENDS.get(package, set()) | {"bootstrap.py"}:
            return
        uses_pymongo = any(
            name.split(".")[0] in {"pymongo", "bson"} for name in imported_names(module)
        )
        assert not uses_pymongo, f"{module_id(module)} imports pymongo"


class TestGivenRoleModules:
    """A role reads the stores the steps write, through its own views, and never writes."""

    def test_when_roles_listed_then_every_role_present(self) -> None:
        assert all((SERVICE / role).is_dir() for role in ROLES)

    @pytest.mark.parametrize("module", ROLE_MODULES, ids=module_id)
    def test_when_in_role_then_no_index_creation_or_write(self, module: Path) -> None:
        accesses = {access.split(".", 1)[1] for access in attribute_accesses(module)}
        writes = {"create_index", "insert_one", "insert_many", "update_one", "update_many"}
        writes |= {"delete_one", "delete_many", "replace_one", "find_one_and_update", "drop"}
        offending = accesses & writes
        assert not offending, f"{module_id(module)} calls {sorted(offending)}"


def top_level_imports(module: Path) -> set[str]:
    """Names imported at module level, outside any function or class body."""
    names: set[str] = set()
    for node in ast.parse(module.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


class TestGivenCommandLine:
    """The entry point loads a step or role, and its extra, only when dispatching to it."""

    def test_when_module_loaded_then_no_package_imported(self) -> None:
        step_packages = {f"scene_rag.extraction.{package}" for package in PACKAGES}
        offending = {
            name
            for name in top_level_imports(SERVICE / "__main__.py")
            if any(name.startswith(package) for package in step_packages)
        }
        assert not offending, f"__main__ imports {sorted(offending)} at module level"


class TestGivenServiceModules:
    """Boundaries across steps and roles."""

    @pytest.mark.parametrize("module", ALL_MODULES, ids=module_id)
    def test_when_outside_step_then_no_writer_side_import(self, module: Path) -> None:
        names = imported_names(module)
        for step, forbidden in WRITER_SIDE.items():
            if package_of(module) == step:
                continue
            offending = {
                name
                for name in names
                if any(name == f or name.startswith(f + ".") for f in forbidden)
            }
            assert not offending, (
                f"{module_id(module)} imports {sorted(offending)}: go through {step}'s bootstrap"
            )

    @pytest.mark.parametrize("module", ALL_MODULES, ids=module_id)
    def test_when_not_designated_backend_then_no_confined_tool_import(self, module: Path) -> None:
        names = imported_names(module)
        for tool, allowed in CONFINED_TOOLS.items():
            if module_id(module) in allowed:
                continue
            offending = {n for n in names if n == tool or n.startswith(tool + ".")}
            assert not offending, f"{module_id(module)} imports {tool}; only {sorted(allowed)} may"
