"""Unit tests for OntologyGraph — in-memory mock backend, no FalkorDB needed."""

import pytest

from smartmemory.graph.ontology_graph import OntologyGraph


class MockBackend:
    """In-memory mock that mimics FalkorDB query() for EntityType nodes."""

    def __init__(self):
        self._nodes: dict[str, str] = {}  # name -> status

    def query(self, cypher: str, params=None, graph_name=None):
        """Parse simple CREATE/MATCH patterns to simulate graph operations."""
        params = params or {}

        if "CREATE" in cypher:
            name = params["name"]
            status = params["status"]
            self._nodes[name] = status
            return []

        if "SET" in cypher:
            name = params["name"]
            if name in self._nodes:
                self._nodes[name] = "confirmed"
                return [[name, "confirmed"]]
            return []

        if "MATCH" in cypher and "RETURN" in cypher:
            if params and "name" in params:
                name = params["name"]
                if name in self._nodes:
                    return [[name, self._nodes[name]]]
                return []
            # Return all, sorted by name
            return sorted([[n, s] for n, s in self._nodes.items()])

        return []


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def backend():
    return MockBackend()


@pytest.fixture
def graph(backend):
    return OntologyGraph(workspace_id="test", backend=backend)


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


def test_seed_types_creates_14_types(graph):
    """seed_types() returns 14 and get_entity_types() reflects all 14."""
    created = graph.seed_types()
    assert created == 14

    types = graph.get_entity_types()
    assert len(types) == 14


def test_seed_types_idempotent(graph):
    """Calling seed_types() twice: second call returns 0."""
    first = graph.seed_types()
    second = graph.seed_types()

    assert first == 14
    assert second == 0


def test_seed_types_custom_list(graph):
    """seed_types() with a custom list seeds only those types."""
    created = graph.seed_types(["Custom1", "Custom2"])

    assert created == 2
    types = graph.get_entity_types()
    assert len(types) == 2
    names = {t["name"] for t in types}
    assert names == {"Custom1", "Custom2"}


def test_all_seed_types_have_seed_status(graph):
    """Every seeded type should have status 'seed'."""
    graph.seed_types()

    types = graph.get_entity_types()
    for t in types:
        assert t["status"] == "seed", f"{t['name']} has status '{t['status']}'"


def test_add_provisional_creates_new(graph):
    """add_provisional() returns True for a brand-new type."""
    result = graph.add_provisional("NewType")
    assert result is True


def test_add_provisional_existing_returns_false(graph):
    """add_provisional() returns False when the type already exists."""
    graph.seed_types()
    result = graph.add_provisional("Person")
    assert result is False


def test_provisional_has_correct_status(graph):
    """A provisionally added type has status 'provisional'."""
    graph.add_provisional("NewType")

    status = graph.get_type_status("NewType")
    assert status == "provisional"


def test_promote_changes_status(graph):
    """promote() sets a type's status to 'confirmed'."""
    graph.seed_types()
    graph.promote("Person")

    status = graph.get_type_status("Person")
    assert status == "confirmed"


def test_get_type_status_unknown_returns_none(graph):
    """get_type_status() returns None for a non-existent type."""
    status = graph.get_type_status("Unknown")
    assert status is None


def test_graph_name_includes_workspace_id():
    """Graph name must be ws_{workspace_id}_ontology."""
    og = OntologyGraph("acme")
    assert og._graph_name == "ws_acme_ontology"
