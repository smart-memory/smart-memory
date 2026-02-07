"""Unit tests for graph index utilities."""

from unittest.mock import MagicMock

from smartmemory.graph.indexes import ensure_extraction_indexes


class TestEnsureExtractionIndexes:
    def test_creates_index(self):
        """Verify the Cypher CREATE INDEX query is executed."""
        backend = MagicMock()

        ensure_extraction_indexes(backend)

        backend.query.assert_called_once()
        query = backend.query.call_args[0][0]
        assert "extraction_status" in query
        assert "CREATE INDEX" in query

    def test_idempotent_no_error_if_exists(self):
        """Verify no error is raised if the index already exists."""
        backend = MagicMock()
        backend.query.side_effect = Exception("Index already exists")

        ensure_extraction_indexes(backend)
        # Should not raise
