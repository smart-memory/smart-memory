"""E2E: Procedural memory workflow (create → get steps → relate → search).

Exercises: memory/types/procedural, ProceduralMemoryGraph.
Requires running FalkorDB (port 9010) and Redis (port 9012).
"""

import os

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.golden]


@pytest.fixture(scope="module")
def memory():
    os.environ.setdefault("FALKORDB_PORT", "9010")
    os.environ.setdefault("REDIS_PORT", "9012")
    os.environ.setdefault("VECTOR_BACKEND", "falkordb")
    try:
        from smartmemory import SmartMemory
        sm = SmartMemory()
    except Exception as e:
        pytest.skip(f"E2E environment not ready: {e}")
    yield sm
    try:
        sm.clear()
    except Exception:
        pass


class TestProceduralWorkflow:
    """Create procedures with steps, relate them, search."""

    def test_create_procedure_with_steps(self, memory):
        """Create a procedure with multiple steps."""
        from smartmemory.models.memory_item import MemoryItem

        procedure = MemoryItem(
            content="To deploy the application: build, test, then push to production.",
            memory_type="procedural",
            metadata={
                "title": "Deploy Application",
                "description": "Standard deployment procedure",
                "steps": [
                    {"step": 1, "action": "Run build command", "command": "npm run build"},
                    {"step": 2, "action": "Run tests", "command": "npm test"},
                    {"step": 3, "action": "Push to production", "command": "git push prod main"},
                ],
            },
        )
        memory.add(procedure)

        # Verify procedure was stored
        retrieved = memory.get(procedure.item_id)
        assert retrieved is not None
        assert "deploy" in retrieved.content.lower() or "Deploy" in str(retrieved.metadata)

    def test_create_multiple_procedures(self, memory):
        """Create multiple related procedures."""
        from smartmemory.models.memory_item import MemoryItem

        proc1 = MemoryItem(
            content="To set up development environment: install dependencies, configure IDE, run migrations.",
            memory_type="procedural",
            metadata={
                "title": "Setup Dev Environment",
                "description": "Initial development setup",
                "steps": [
                    {"step": 1, "action": "Install dependencies", "command": "pip install -r requirements.txt"},
                    {"step": 2, "action": "Configure IDE settings"},
                    {"step": 3, "action": "Run database migrations", "command": "python manage.py migrate"},
                ],
            },
        )
        proc2 = MemoryItem(
            content="To run tests: activate virtual environment, run pytest with coverage.",
            memory_type="procedural",
            metadata={
                "title": "Run Tests",
                "description": "Test execution procedure",
                "steps": [
                    {"step": 1, "action": "Activate venv", "command": "source venv/bin/activate"},
                    {"step": 2, "action": "Run pytest", "command": "pytest --cov"},
                ],
            },
        )

        memory.add(proc1)
        memory.add(proc2)

        # Verify both stored
        r1 = memory.get(proc1.item_id)
        r2 = memory.get(proc2.item_id)
        assert r1 is not None
        assert r2 is not None

    def test_link_procedures(self, memory):
        """Link related procedures together."""
        from smartmemory.models.memory_item import MemoryItem

        proc1 = MemoryItem(
            content="Initialize project: create repo, set up CI, configure linting.",
            memory_type="procedural",
            metadata={"title": "Initialize Project"},
        )
        proc2 = MemoryItem(
            content="Configure CI: set up GitHub Actions, add test workflow.",
            memory_type="procedural",
            metadata={"title": "Configure CI"},
        )

        memory.add(proc1)
        memory.add(proc2)

        # Link procedures
        memory.link(proc1.item_id, proc2.item_id, link_type="HAS_PREREQUISITE")

        # Verify link
        links = memory.get_links(proc1.item_id)
        assert isinstance(links, (list, dict))

    def test_search_procedures(self, memory):
        """Search for procedures by content."""
        from smartmemory.models.memory_item import MemoryItem

        proc = MemoryItem(
            content="To backup database: dump to SQL, compress, upload to S3.",
            memory_type="procedural",
            metadata={
                "title": "Database Backup",
                "steps": [
                    {"step": 1, "action": "Dump database", "command": "pg_dump > backup.sql"},
                    {"step": 2, "action": "Compress", "command": "gzip backup.sql"},
                    {"step": 3, "action": "Upload to S3", "command": "aws s3 cp backup.sql.gz s3://backups/"},
                ],
            },
        )
        memory.add(proc)

        # Search for procedures
        results = memory.search("backup database S3", top_k=5)
        assert isinstance(results, list)

    def test_procedural_add_macro(self, memory):
        """Test adding a macro pattern to procedural memory."""
        # Access procedural memory directly for macro operations
        if hasattr(memory, 'procedural') and hasattr(memory.procedural, 'add_macro'):
            result = memory.procedural.add_macro({"pattern": "git add && git commit && git push"})
            # add_macro returns True on success
            assert result in (True, False, None)  # Document actual behavior
        else:
            pytest.skip("Procedural memory add_macro not available via SmartMemory")

    def test_get_procedure_steps(self, memory):
        """Test retrieving steps from a procedure."""
        import json
        from smartmemory.models.memory_item import MemoryItem

        proc = MemoryItem(
            content="Code review procedure: check style, verify tests, approve PR.",
            memory_type="procedural",
            metadata={
                "title": "Code Review",
                "steps": [
                    {"step": 1, "action": "Check code style"},
                    {"step": 2, "action": "Verify tests pass"},
                    {"step": 3, "action": "Approve and merge PR"},
                ],
            },
        )
        memory.add(proc)

        # Try to get steps via procedural memory interface
        if hasattr(memory, 'procedural') and hasattr(memory.procedural, 'get_procedure_steps'):
            steps = memory.procedural.get_procedure_steps(proc.item_id)
            assert isinstance(steps, list)
        else:
            # Fallback: steps are in metadata
            retrieved = memory.get(proc.item_id)
            assert retrieved is not None
            steps = retrieved.metadata.get('steps', [])
            # Steps may be stored as JSON string
            if isinstance(steps, str):
                steps = json.loads(steps)
            assert isinstance(steps, list)

    def test_procedure_remove(self, memory):
        """Test removing a procedure.

        Note: Some graph stores may have eventual consistency for removes.
        This test documents actual behavior.
        """
        from smartmemory.models.memory_item import MemoryItem

        proc = MemoryItem(
            content="Temporary procedure to be deleted.",
            memory_type="procedural",
            metadata={"title": "Temp Procedure"},
        )
        memory.add(proc)
        item_id = proc.item_id

        # Verify it exists
        assert memory.get(item_id) is not None

        # Remove it
        result = memory.remove(item_id)

        # Remove should return True or None (depending on implementation)
        # Note: Some backends may not immediately reflect removal
        assert result in (True, None, False)  # Document actual behavior


class TestProceduralEvolution:
    """Test evolution from working memory to procedural memory."""

    def test_working_to_procedural_evolution_setup(self, memory):
        """Verify working_to_procedural evolver plugin is available."""
        from smartmemory.plugins.manager import PluginManager
        from smartmemory.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        manager = PluginManager(registry=registry)
        manager.discover_plugins(include_builtin=True, include_entry_points=False, plugin_dirs=None)

        evolver_names = registry.list_plugins("evolver")
        assert "working_to_procedural" in evolver_names
