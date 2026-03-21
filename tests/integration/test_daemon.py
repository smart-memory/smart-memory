"""Integration tests for DIST-DAEMON-1 Task 11: daemon lifecycle.

These tests exercise the real daemon start/stop/health cycle using subprocess
and actual port binding. They are auto-marked @integration by conftest.py.

Requirements: no external services — just Python + localhost ports.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Use a non-default port to avoid conflicting with a user's running daemon
TEST_PORT = 19014


def _port_open(port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _wait_for_port(port: int, timeout: float = 60.0) -> bool:
    """Wait until a port is open or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port, timeout=0.5):
            return True
        time.sleep(0.5)
    return False


def _wait_for_port_closed(port: int, timeout: float = 10.0) -> bool:
    """Wait until a port is closed or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_open(port, timeout=0.5):
            return True
        time.sleep(0.5)
    return False


@pytest.fixture()
def daemon_env(tmp_path, monkeypatch):
    """Set up isolated environment for daemon tests."""
    data_dir = tmp_path / ".smartmemory"
    data_dir.mkdir()
    config_dir = tmp_path / ".config" / "smartmemory"
    config_dir.mkdir(parents=True)

    # Write a minimal config
    config_file = config_dir / "config.toml"
    config_file.write_text(f"""
[smartmemory]
mode = "local"

[local]
llm_provider = "none"
embedding_provider = "local"
daemon_port = {TEST_PORT}
data_dir = "{data_dir}"
""")

    env = os.environ.copy()
    env["SMARTMEMORY_DATA_DIR"] = str(data_dir)
    env["SMARTMEMORY_DAEMON_PORT"] = str(TEST_PORT)
    env["XDG_CONFIG_HOME"] = str(tmp_path / ".config")

    yield {"data_dir": data_dir, "config_dir": config_dir, "env": env, "port": TEST_PORT}

    # Cleanup: kill any daemon left on the test port
    if _port_open(TEST_PORT, timeout=0.5):
        try:
            import httpx
            r = httpx.get(f"http://127.0.0.1:{TEST_PORT}/health", timeout=2)
            pid = r.json().get("pid")
            if pid:
                os.kill(pid, signal.SIGTERM)
                _wait_for_port_closed(TEST_PORT, timeout=5)
        except Exception:
            pass


class TestDaemonHelpers:
    """Unit-level tests for daemon.py helpers that don't need a running daemon."""

    def test_is_running_false_when_nothing_on_port(self, daemon_env, monkeypatch):
        monkeypatch.setenv("SMARTMEMORY_DAEMON_PORT", str(daemon_env["port"]))
        monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(daemon_env["data_dir"]))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(daemon_env["config_dir"].parent))

        from smartmemory_app.daemon import is_running
        assert is_running() is False
        assert is_running(require_healthy=False) is False

    def test_get_status_none_when_not_running(self, daemon_env, monkeypatch):
        monkeypatch.setenv("SMARTMEMORY_DAEMON_PORT", str(daemon_env["port"]))
        monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(daemon_env["data_dir"]))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(daemon_env["config_dir"].parent))

        from smartmemory_app.daemon import get_status
        assert get_status() is None

    def test_stop_daemon_noop_when_not_running(self, daemon_env, monkeypatch):
        """stop_daemon() should not raise when daemon is not running."""
        monkeypatch.setenv("SMARTMEMORY_DAEMON_PORT", str(daemon_env["port"]))
        monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(daemon_env["data_dir"]))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(daemon_env["config_dir"].parent))

        from smartmemory_app.daemon import stop_daemon
        stop_daemon()  # should not raise


class TestDaemonIsRunningDistinguishesProcesses:
    """is_running() must return False when a non-SmartMemory process holds the port."""

    def test_non_smartmemory_on_port_returns_false(self, daemon_env, monkeypatch):
        """Start a dummy HTTP server on the test port — is_running() must return False."""
        monkeypatch.setenv("SMARTMEMORY_DAEMON_PORT", str(daemon_env["port"]))
        monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(daemon_env["data_dir"]))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(daemon_env["config_dir"].parent))

        # Start a minimal HTTP server that returns non-SmartMemory JSON on /health
        server_code = f"""
import http.server, json, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({{"service": "other-app"}}).encode())
    def log_message(self, *a): pass
with socketserver.TCPServer(("127.0.0.1", {daemon_env["port"]}), H) as s:
    s.serve_forever()
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", server_code],
            start_new_session=True,
        )
        try:
            assert _wait_for_port(daemon_env["port"], timeout=5), "Dummy server should bind"

            from smartmemory_app.daemon import is_running
            assert is_running() is False, "is_running() must return False for non-SmartMemory process"
            assert is_running(require_healthy=False) is False
        finally:
            proc.terminate()
            proc.wait(timeout=5)


class TestHealthEndpointShape:
    """Test the /health endpoint returns expected fields via TestClient (no daemon needed)."""

    def test_health_returns_required_fields(self):
        from smartmemory.graph.backends.sqlite import SQLiteBackend
        import smartmemory_app.local_api as _api_mod
        from smartmemory_app.viewer_server import app
        from fastapi.testclient import TestClient

        backend = SQLiteBackend(db_path=":memory:")
        with patch.object(_api_mod, "_get_backend", return_value=backend):
            client = TestClient(app)
            r = client.get("/health")

        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "smartmemory"
        assert body["status"] in ("ok", "degraded")
        assert "pid" in body
        assert "memories" in body
        assert "llm_provider" in body
        assert "embedding_provider" in body
        assert "async_enrichment" in body
        assert isinstance(body["async_enrichment"], dict)
        assert "enabled" in body["async_enrichment"]
        backend.close()

    def test_health_status_is_degraded_when_backend_fails(self):
        from smartmemory_app.viewer_server import app
        from fastapi.testclient import TestClient

        with patch("smartmemory_app.viewer_server._local_api"):
            client = TestClient(app)
            # Patch get_memory to raise — health should return "degraded"
            with patch("smartmemory_app.storage.get_memory", side_effect=RuntimeError("no backend")):
                r = client.get("/health")

        body = r.json()
        assert body["status"] == "degraded"


class TestCLIDaemonFallback:
    """CLI commands must work without a daemon (fallback to direct storage)."""

    def test_persist_falls_back_when_daemon_down(self, daemon_env, monkeypatch):
        """persist command calls storage.ingest when daemon is unreachable."""
        monkeypatch.setenv("SMARTMEMORY_DAEMON_PORT", str(daemon_env["port"]))

        from click.testing import CliRunner
        from smartmemory_app.cli import cli

        with patch("smartmemory_app.storage.ingest", return_value="fallback-id") as mock:
            runner = CliRunner()
            result = runner.invoke(cli, ["add", "test text"])

        assert result.exit_code == 0
        assert "fallback-id" in result.output
        mock.assert_called_once()

    def test_search_falls_back_when_daemon_down(self, daemon_env, monkeypatch):
        """search command calls storage.search when daemon is unreachable."""
        monkeypatch.setenv("SMARTMEMORY_DAEMON_PORT", str(daemon_env["port"]))

        from click.testing import CliRunner
        from smartmemory_app.cli import cli

        mock_results = [{"item_id": "abc", "content": "test", "memory_type": "semantic"}]
        with patch("smartmemory_app.storage.search", return_value=mock_results) as mock:
            runner = CliRunner()
            result = runner.invoke(cli, ["search", "test query"])

        assert result.exit_code == 0
        assert "abc" in result.output
        mock.assert_called_once()
