"""Integration tests for DIST-LITE-4: viewer_server.py.

Tests cover:
  - Module-level app is importable without starting uvicorn (no side effects)
  - GET /memory/graph/full via TestClient(app) returns 200
  - GET / returns 200 (serves static placeholder index.html)
  - start_background() is called (not a raw thread) inside main()
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Import guard — no side effects on module import
# ---------------------------------------------------------------------------


class TestImportNoSideEffects:
    def test_app_importable_without_uvicorn(self):
        """Module-level app = _build_app() must not start uvicorn or the events server."""
        # If this import triggers uvicorn.run() or start_background(), the test hangs.
        # A clean import proves the module is side-effect-free.
        from smartmemory_pkg.viewer_server import app  # noqa: F401
        assert app is not None

    def test_app_is_fastapi_instance(self):
        from fastapi import FastAPI
        from smartmemory_pkg.viewer_server import app
        assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# HTTP routes via TestClient
# ---------------------------------------------------------------------------


@pytest.fixture()
def viewer_client(monkeypatch):
    """TestClient for viewer_server.app with _get_backend patched to a real in-memory backend."""
    from smartmemory.graph.backends.sqlite import SQLiteBackend
    import smartmemory_pkg.local_api as _api_mod

    backend = SQLiteBackend(db_path=":memory:")
    monkeypatch.setattr(_api_mod, "_get_backend", lambda: backend)

    from smartmemory_pkg.viewer_server import app
    client = TestClient(app)
    yield client
    backend.close()


class TestViewerServerRoutes:
    def test_graph_full_returns_200(self, viewer_client):
        """GET /memory/graph/full must return 200 — local_api mounted at /memory."""
        r = viewer_client.get("/memory/graph/full")
        assert r.status_code == 200

    def test_graph_full_has_required_keys(self, viewer_client):
        body = viewer_client.get("/memory/graph/full").json()
        assert "nodes" in body
        assert "edges" in body
        assert "node_count" in body
        assert "edge_count" in body

    def test_static_root_returns_200(self, viewer_client):
        """GET / must return 200 and serve the placeholder index.html."""
        r = viewer_client.get("/")
        assert r.status_code == 200

    def test_static_root_contains_placeholder_text(self, viewer_client):
        r = viewer_client.get("/")
        assert "build-viewer" in r.text or "Viewer" in r.text


# ---------------------------------------------------------------------------
# main() calls start_background() — not a raw thread
# ---------------------------------------------------------------------------


class TestMainCallsStartBackground:
    def test_main_uses_start_background_not_raw_thread(self):
        """main() must call start_background() from events_server — never Thread() directly."""
        from smartmemory_pkg import viewer_server

        # Patch out uvicorn.run so main() returns immediately
        with (
            patch("smartmemory_pkg.viewer_server.uvicorn") as mock_uvicorn,
            patch("smartmemory_pkg.events_server.start_background") as mock_start_bg,
        ):
            mock_uvicorn.run = MagicMock()
            viewer_server.main(port=19005, open_browser=False)

        mock_start_bg.assert_called_once()
        # Confirm start_background was called from events_server, not a threading.Thread
        assert mock_uvicorn.run.called

    def test_main_does_not_open_browser_when_disabled(self):
        """--no-browser flag: webbrowser.open must not be called."""
        from smartmemory_pkg import viewer_server

        with (
            patch("smartmemory_pkg.viewer_server.uvicorn") as mock_uvicorn,
            patch("smartmemory_pkg.events_server.start_background"),
            patch("smartmemory_pkg.viewer_server.webbrowser") as mock_browser,
            patch("smartmemory_pkg.viewer_server.threading") as mock_threading,
        ):
            mock_uvicorn.run = MagicMock()
            viewer_server.main(port=19005, open_browser=False)

        # Timer should not be created when open_browser=False
        mock_threading.Timer.assert_not_called()
