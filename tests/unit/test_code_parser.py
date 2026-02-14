"""Tests for the Code Connector AST parser and indexer.

Tests parsing of Python files into code entities and relationships.
Uses a fixture file to validate extraction of classes, functions, imports,
routes, tests, and various relationship types. Also tests CodeIndexer
orchestration with a mock graph.
"""

import os
from unittest.mock import MagicMock

import pytest

from smartmemory.code.models import CodeEntity, IndexResult
from smartmemory.code.indexer import CodeIndexer
from smartmemory.code.parser import CodeParser, collect_python_files


# -- Fixtures --

SAMPLE_PYTHON_CODE = '''
"""A sample module for testing the code parser."""

import os
from typing import Optional, List

from fastapi import APIRouter
from some_module import helper_function


router = APIRouter()


class BaseModel:
    """A base model class."""
    pass


class UserService(BaseModel):
    """Service for managing users."""

    def create_user(self, name: str) -> dict:
        """Create a new user."""
        result = helper_function(name)
        return {"name": name, "result": result}

    def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID."""
        return True


@router.get("/users")
def list_users(limit: int = 10) -> list:
    """List all users."""
    service = UserService()
    return service.create_user("test")


@router.post("/users")
async def create_user_endpoint(name: str) -> dict:
    """Create a user via API."""
    service = UserService()
    return service.create_user(name)


def helper_function(x):
    """A helper that does something."""
    return x


class TestUserService:
    """Tests for UserService."""

    def test_create_user(self):
        """Test user creation."""
        svc = UserService()
        result = svc.create_user("alice")
        assert result["name"] == "alice"

    def test_delete_user(self):
        """Test user deletion."""
        svc = UserService()
        assert svc.delete_user("123") is True
'''


@pytest.fixture
def sample_file(tmp_path):
    """Create a temporary Python file with sample code."""
    file_path = tmp_path / "sample_service.py"
    file_path.write_text(SAMPLE_PYTHON_CODE)
    return str(file_path), str(tmp_path)


@pytest.fixture
def parser(sample_file):
    """Create a parser for the sample file."""
    _, repo_root = sample_file
    return CodeParser(repo="test-repo", repo_root=repo_root)


# -- Model Tests --


class TestCodeEntity:
    def test_item_id_deterministic(self):
        entity = CodeEntity(
            name="MyClass",
            entity_type="class",
            file_path="src/models.py",
            line_number=10,
            repo="my-repo",
        )
        assert entity.item_id == "code::my-repo::src/models.py::MyClass"

    def test_to_properties(self):
        entity = CodeEntity(
            name="get_users",
            entity_type="route",
            file_path="routes/users.py",
            line_number=42,
            repo="api",
            docstring="Get all users",
            http_method="GET",
            http_path="/users",
            decorators=["router.get"],
            bases=[],
        )
        props = entity.to_properties()
        assert props["item_id"] == "code::api::routes/users.py::get_users"
        assert props["name"] == "get_users"
        assert props["entity_type"] == "route"
        assert props["memory_type"] == "code"
        assert props["http_method"] == "GET"
        assert props["http_path"] == "/users"
        assert props["decorators"] == "router.get"

    def test_to_properties_omits_empty_optional_fields(self):
        entity = CodeEntity(
            name="foo",
            entity_type="function",
            file_path="a.py",
            line_number=1,
            repo="r",
        )
        props = entity.to_properties()
        assert "docstring" not in props
        assert "decorators" not in props
        assert "bases" not in props
        assert "http_method" not in props


class TestIndexResult:
    def test_defaults(self):
        result = IndexResult(repo="my-repo")
        assert result.files_parsed == 0
        assert result.entities_created == 0
        assert result.edges_created == 0
        assert result.errors == []


# -- Parser Tests --


class TestCodeParser:
    def test_parse_module_entity(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        modules = [e for e in result.entities if e.entity_type == "module"]
        assert len(modules) == 1
        assert modules[0].name == "sample_service"
        assert modules[0].line_number == 1

    def test_parse_classes(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        classes = [e for e in result.entities if e.entity_type == "class"]
        class_names = {c.name for c in classes}
        assert "BaseModel" in class_names
        assert "UserService" in class_names

    def test_parse_functions(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        functions = [e for e in result.entities if e.entity_type == "function"]
        func_names = {f.name for f in functions}
        assert "helper_function" in func_names
        assert "UserService.create_user" in func_names
        assert "UserService.delete_user" in func_names

    def test_parse_routes(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        routes = [e for e in result.entities if e.entity_type == "route"]
        assert len(routes) == 2

        get_route = next(r for r in routes if r.http_method == "GET")
        assert get_route.name == "list_users"
        assert get_route.http_path == "/users"

        post_route = next(r for r in routes if r.http_method == "POST")
        assert post_route.name == "create_user_endpoint"
        assert post_route.http_path == "/users"

    def test_parse_tests(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        tests = [e for e in result.entities if e.entity_type == "test"]
        test_names = {t.name for t in tests}
        assert "TestUserService" in test_names
        assert "TestUserService.test_create_user" in test_names
        assert "TestUserService.test_delete_user" in test_names

    def test_parse_docstrings(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        user_service = next(e for e in result.entities if e.name == "UserService")
        assert user_service.docstring == "Service for managing users."

    def test_parse_inheritance(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        inherits = [r for r in result.relations if r.relation_type == "INHERITS"]
        assert len(inherits) >= 1
        # UserService inherits BaseModel
        us_inherits = [r for r in inherits if "UserService" in r.source_id and "BaseModel" in r.target_id]
        assert len(us_inherits) == 1

    def test_parse_defines_edges(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        defines = [r for r in result.relations if r.relation_type == "DEFINES"]
        # Module should define classes + top-level functions
        assert (
            len(defines) >= 5
        )  # BaseModel, UserService, list_users, create_user_endpoint, helper_function, TestUserService + methods

    def test_parse_imports(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        imports = [r for r in result.relations if r.relation_type == "IMPORTS"]
        assert len(imports) >= 3  # os, typing, fastapi, some_module

    def test_parse_calls(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        calls = [r for r in result.relations if r.relation_type == "CALLS"]
        # Functions call helper_function, UserService, create_user, etc.
        assert len(calls) >= 1

    def test_parse_tests_edges(self, parser, sample_file):
        """Test that TESTS edges link test methods to tested functions."""
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        tests_edges = [r for r in result.relations if r.relation_type == "TESTS"]
        assert len(tests_edges) >= 1
        # test_create_user should link to UserService.create_user
        test_targets = {r.target_id for r in tests_edges}
        assert any("create_user" in t for t in test_targets)

    def test_parse_tests_edges_class_qualified(self, tmp_path):
        """Test that class-qualified test methods (TestFoo.test_bar) create TESTS edges."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(1, 2) == 3
"""
        file_path = tmp_path / "test_calc.py"
        file_path.write_text(code)
        parser = CodeParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(file_path))

        tests_edges = [r for r in result.relations if r.relation_type == "TESTS"]
        assert len(tests_edges) == 1
        # TestCalculator.test_add should link to Calculator.add
        assert "Calculator.add" in tests_edges[0].target_id

    def test_handles_syntax_error(self, parser, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:\n    pass")
        result = parser.parse_file(str(bad_file))
        assert len(result.errors) == 1
        assert "SyntaxError" in result.errors[0]

    def test_handles_empty_file(self, parser, tmp_path):
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        result = parser.parse_file(str(empty_file))
        # Should still create a module entity
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "module"

    def test_init_py_module_name(self, tmp_path):
        """__init__.py files should use package name, not include .__init__ suffix."""
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        init_file = pkg / "__init__.py"
        init_file.write_text('"""Package docstring."""\n')
        parser = CodeParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(init_file))

        modules = [e for e in result.entities if e.entity_type == "module"]
        assert len(modules) == 1
        assert modules[0].name == "mypkg"  # NOT mypkg.__init__

    def test_defines_edges_class_owns_methods(self, parser, sample_file):
        """DEFINES edges for methods should source from the class, not the module."""
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        # Find the DEFINES edge for UserService.create_user specifically
        create_user_defines = [
            r
            for r in result.relations
            if r.relation_type == "DEFINES" and r.target_id.endswith("::UserService.create_user")
        ]
        assert len(create_user_defines) == 1
        # Source should be UserService (class), not the module
        assert create_user_defines[0].source_id.endswith("::UserService")


# -- File Collection Tests --


class TestCollectPythonFiles:
    def test_collects_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "c.txt").write_text("not python")

        files = collect_python_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_excludes_directories(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("y = 2")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "hooks.py").write_text("z = 3")

        files = collect_python_files(str(tmp_path))
        file_names = [os.path.basename(f) for f in files]
        assert "app.py" in file_names
        assert "cached.py" not in file_names
        assert "hooks.py" not in file_names

    def test_recurses_into_subdirectories(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "deep.py").write_text("x = 1")

        files = collect_python_files(str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("deep.py")


# -- Indexer Tests --


class TestCodeIndexer:
    """Tests for CodeIndexer orchestration with a mock graph."""

    def test_golden_flow(self, tmp_path):
        """Index a small directory and verify bulk graph calls."""
        (tmp_path / "app.py").write_text('class App:\n    """Main app."""\n    pass\n')
        (tmp_path / "utils.py").write_text("def helper():\n    return 1\n")

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 4  # module + class + module + function
        graph.add_edges_bulk.return_value = 2  # DEFINES edges

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.files_parsed == 2
        assert result.entities_created > 0
        assert result.edges_created >= 0
        assert result.errors == []
        # Verify delete was called
        graph.execute_query.assert_called_once()
        assert "DETACH DELETE" in graph.execute_query.call_args[0][0]
        # Verify bulk methods were called once each
        graph.add_nodes_bulk.assert_called_once()
        graph.add_edges_bulk.assert_called_once()
        # Verify the bulk node list contains properties dicts
        bulk_nodes = graph.add_nodes_bulk.call_args[0][0]
        assert all(isinstance(n, dict) and "item_id" in n for n in bulk_nodes)

    def test_skips_dangling_edges(self, tmp_path):
        """Edges referencing non-existent entities are filtered out."""
        code = "from external_lib import something\ndef foo():\n    something()\n"
        (tmp_path / "caller.py").write_text(code)

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 2
        graph.add_edges_bulk.return_value = 1

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        result = indexer.index()

        # The bulk edge list should only contain edges where both endpoints exist
        bulk_edges = graph.add_edges_bulk.call_args[0][0]
        entity_ids = {n["item_id"] for n in graph.add_nodes_bulk.call_args[0][0]}
        for source_id, target_id, edge_type, _ in bulk_edges:
            assert source_id in entity_ids and target_id in entity_ids

    def test_handles_graph_write_errors(self, tmp_path):
        """Graph write failures are captured as errors, not raised."""
        (tmp_path / "simple.py").write_text("x = 1\n")

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.side_effect = RuntimeError("Graph unavailable")
        graph.add_edges_bulk.return_value = 0

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.entities_created == 0
        assert len(result.errors) > 0
        assert "Graph unavailable" in result.errors[0]

    def test_delete_includes_workspace_scope(self, tmp_path):
        """When scope_provider has workspace_id, delete query should include it."""
        (tmp_path / "app.py").write_text("x = 1\n")

        # Mock graph with scope_provider returning workspace_id
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws-123"}
        backend = MagicMock()
        backend.scope_provider = scope_provider

        graph = MagicMock(spec=["execute_query", "add_nodes_bulk", "add_edges_bulk", "nodes"])
        graph.nodes.backend = backend
        graph.execute_query.return_value = []

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        indexer.index()

        # The delete query should contain workspace_id filter
        delete_call = graph.execute_query.call_args
        query_str = delete_call[0][0]
        params = delete_call[0][1]
        assert "workspace_id" in query_str
        assert params["workspace_id"] == "ws-123"
        assert "DETACH DELETE" in query_str
