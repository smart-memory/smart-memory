"""DIST-DAEMON-1: Daemon lifecycle helpers — start/stop the viewer server as a background process.

Not a server itself — just functions for managing the daemon process.
The daemon IS viewer_server.main() running in a detached subprocess.
"""
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _data_dir() -> Path:
    """Resolve data dir from config (respects data_dir setting and SMARTMEMORY_DATA_DIR env)."""
    from smartmemory_app.storage import _resolve_data_dir
    return _resolve_data_dir()


def _pid_file() -> Path:
    return _data_dir() / "daemon.pid"


def _port() -> int:
    from smartmemory_app.config import load_config
    return load_config().daemon_port


def is_running(require_healthy: bool = True) -> bool:
    """Check daemon is running AND is SmartMemory (not a random process on the port).

    require_healthy=True: also checks backend loaded ("ok" status).
    require_healthy=False: any SmartMemory response counts (for stop/status).
    """
    try:
        import httpx
        r = httpx.get(f"http://127.0.0.1:{_port()}/health", timeout=2)
        data = r.json()
        if data.get("service") != "smartmemory":
            return False
        if require_healthy and data.get("status") != "ok":
            return False
        return True
    except Exception:
        return False


def start_daemon(num_workers: int = 1) -> None:
    """Start the daemon and enrichment workers.

    Blocks until daemon is ready (health check passes) or timeout.
    Then starts num_workers background enrichment worker processes.

    Warmup takes ~22s cold (first run), ~2s warm (model cached).
    Idempotent — returns immediately if already running.
    """
    if is_running():
        return

    port = _port()
    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)
    log_path = data / "daemon.log"

    # Launch viewer_server.main() directly — NOT the CLI command
    # (avoids recursion since CLI `viewer` calls start_daemon + open browser).
    # Inherit PYTHONPATH so editable installs work in dev.
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"from smartmemory_app.viewer_server import main; main(port={port}, open_browser=False)"],
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    # Phase 1: Wait for port to open (fast socket check, no httpx timeout)
    import socket
    for _ in range(120):  # 60s max
        if proc.poll() is not None:
            raise RuntimeError(
                f"Daemon exited during startup (code {proc.returncode}). Check {log_path}"
            )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break  # port is open
        finally:
            s.close()
        time.sleep(0.5)
    else:
        proc.terminate()
        raise TimeoutError(f"Daemon failed to bind port within 60s. Check {log_path}")

    # Phase 2: Verify it's actually SmartMemory responding
    if not is_running(require_healthy=False):
        proc.terminate()
        raise RuntimeError(f"Port {port} is open but health check failed. Check {log_path}")

    # Phase 3: Start enrichment worker(s)
    _start_workers(num_workers)


def _start_workers(num_workers: int = 1) -> None:
    """Start enrichment worker(s) as detached background processes.

    Each worker polls the SQLite enrichment_queue and runs Tier 2 LLM extraction.
    Multiple workers process jobs in parallel (SQLite row-level locking prevents
    double-processing via the 'processing' status).

    Args:
        num_workers: Number of worker processes to start (default 1).
    """
    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return

    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)
    worker_log = data / "worker.log"
    env = os.environ.copy()

    for i in range(num_workers):
        pid_file = data / f"worker.{i}.pid"

        # Check if this slot is already running
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                continue  # already running
            except (ProcessLookupError, ValueError):
                pid_file.unlink(missing_ok=True)

        proc = subprocess.Popen(
            [sys.executable, "-m", "smartmemory_app.enrichment_worker", "--loop"],
            stdout=open(worker_log, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        pid_file.write_text(str(proc.pid))


def _stop_workers() -> None:
    """Stop all enrichment workers. Idempotent."""
    data = _data_dir()
    for pid_file in data.glob("worker.*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        pid_file.unlink(missing_ok=True)


def stop_daemon() -> None:
    """Stop the daemon and all workers. Idempotent — no-op if not running."""
    _stop_workers()
    import httpx

    # Prefer health-check-based stop — confirms we're killing SmartMemory, not a reused PID
    if is_running(require_healthy=False):
        try:
            r = httpx.get(f"http://127.0.0.1:{_port()}/health", timeout=2)
            pid = r.json().get("pid")
            if pid:
                os.kill(pid, signal.SIGTERM)
                for _ in range(20):
                    if not is_running(require_healthy=False):
                        _pid_file().unlink(missing_ok=True)
                        return
                    time.sleep(0.25)
                # Still running after 5s — force kill
                os.kill(pid, signal.SIGKILL)
                _pid_file().unlink(missing_ok=True)
                return
        except Exception:
            pass

    # Fallback: PID file (only if health unreachable but file exists)
    pf = _pid_file()
    if pf.exists():
        try:
            pid = int(pf.read_text().strip())
            # Verify it's actually a smartmemory process before killing
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True,
            )
            if "smartmemory" in result.stdout:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        pf.unlink(missing_ok=True)


def get_status() -> dict | None:
    """Get daemon status. Returns health dict or None if not running."""
    if not is_running(require_healthy=False):
        return None
    try:
        import httpx
        r = httpx.get(f"http://127.0.0.1:{_port()}/health", timeout=3)
        return r.json()
    except Exception:
        return {"service": "smartmemory", "status": "unreachable"}
