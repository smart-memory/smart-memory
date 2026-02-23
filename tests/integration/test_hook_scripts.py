"""Integration tests for hook shell scripts."""

import json
import subprocess
from pathlib import Path
import pytest

HOOKS_DIR = Path(__file__).parents[2] / "smartmemory_cc" / "hooks"


@pytest.mark.integration
def test_session_start_hook_exit_0(tmp_path, monkeypatch):
    """session-start.sh exits 0 even with no memories stored."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))
    hook = HOOKS_DIR / "session-start.sh"
    payload = json.dumps({"cwd": str(tmp_path)}).encode()

    result = subprocess.run(
        ["bash", str(hook)],
        input=payload,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"session-start.sh must exit 0. stderr: {result.stderr.decode()}"
    )


@pytest.mark.integration
def test_post_tool_failure_skips_interrupt(tmp_path, monkeypatch):
    """post-tool-failure.sh skips ingest when is_interrupt=true."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))
    hook = HOOKS_DIR / "post-tool-failure.sh"
    # is_interrupt=true: hook must exit 0 immediately without calling ingest
    payload = json.dumps(
        {
            "is_interrupt": True,
            "tool_name": "Bash",
            "error": "User interrupted",
        }
    ).encode()

    result = subprocess.run(
        ["bash", str(hook)],
        input=payload,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, "post-tool-failure.sh must exit 0 on interrupt"
    # No ingest should have been called — no output expected
    assert result.stdout == b"", "No stdout expected when skipping due to interrupt"
