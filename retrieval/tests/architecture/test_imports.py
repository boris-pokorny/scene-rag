"""Import rules: step isolation, environment only in bootstrap, tools confined."""

import ast
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[2] / "src" / "scene_rag" / "retrieval"
STEPS = {"chunking", "embedding"}
# A role obeys the step rules and never writes.
ROLES = {"query_api"}
PACKAGES = STEPS | ROLES

# Per package: the modules that may import pymongo, besides bootstrap.
STORE_BACKENDS = {
    "chunking": {"chunk_store.py"},
    "embedding": {"chunk_reader.py", "index_store.py"},
    "query_api": {"index_ledger.py"},
}
# Per step: the write side other steps and roles must never import.
WRITER_SIDE = {
    "chunking": {
        "scene_rag.retrieval.chunking.protocols.ChunkStoreWriter",
        "scene_rag.retrieval.chunking.chunk_store",
    },
    "embedding": {
        "scene_rag.retrieval.embedding.protocols.SearchIndexWriter",
        "scene_rag.retrieval.embedding.protocols.SearchIndex",
        "scene_rag.retrieval.embedding.protocols.IndexStoreWriter",
        "scene_rag.retrieval.embedding.index_store",
        "scene_rag.retrieval.embedding.qdrant_index",
    },
}
# Tools confined to named modules: a package, or a module within one, per tool.
CONFINED_TOOLS = {
    "urllib.request": {
        "chunking/http_transport.py",
        "embedding/http_transport.py",
        "query_api/http_transport.py",
    },
    "fastapi": {"query_api/app.py", "query_api/openapi.py", "query_api/bootstrap.py"},
    "starlette": {"query_api/app.py"},
    "uvicorn": {"query_api/bootstrap.py"},
    "mcp": {
        "query_api/mcp_server.py",
        "query_api/mcp_schema.py",
        "query_api/app.py",
        "query_api/bootstrap.py",
    },
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
        foreign = {f"scene_rag.retrieval.{other}" for other in PACKAGES - {package}}
        offending = {
            name for name in imported_names(module) if any(name.startswith(f) for f in foreign)
        }
        assert not offending, f"{module_id(module)} imports {sorted(offending)}"

    @pytest.mark.parametrize("module", PACKAGE_MODULES, ids=module_id)
    def test_when_in_package_then_no_extraction_import(self, module: Path) -> None:
        offending = {n for n in imported_names(module) if n.startswith("scene_rag.extraction")}
        assert not offending, (
            f"{module_id(module)} imports {sorted(offending)}: Media data comes through scene-api"
        )

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
        writes |= {"upsert", "ensure_collection", "drop_collection", "put_index", "put_index_set"}
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
        step_packages = {f"scene_rag.retrieval.{package}" for package in PACKAGES}
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
