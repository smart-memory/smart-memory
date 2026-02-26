import os


def push_if_configured() -> None:
    """Push local memories to SmartMemory cloud.

    No-op if SMARTMEMORY_SYNC_TOKEN is not set.
    Raises NotImplementedError if token is set — sync not yet implemented.
    """
    if not os.environ.get("SMARTMEMORY_SYNC_TOKEN"):
        return
    raise NotImplementedError(
        "Backend sync not yet implemented. Coming in DIST-PLUGIN-2. "
        "Unset SMARTMEMORY_SYNC_TOKEN to suppress this error."
    )


def pull_if_configured() -> None:
    """Pull memories from SmartMemory cloud.

    No-op if SMARTMEMORY_SYNC_TOKEN is not set.
    """
    if not os.environ.get("SMARTMEMORY_SYNC_TOKEN"):
        return
    raise NotImplementedError(
        "Backend sync not yet implemented. Coming in DIST-PLUGIN-2. "
        "Unset SMARTMEMORY_SYNC_TOKEN to suppress this error."
    )
