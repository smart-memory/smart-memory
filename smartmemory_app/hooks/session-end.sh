#!/usr/bin/env bash
# DIST-DAEMON-1: Persist last assistant message. Daemon (instant) with CLI fallback.
DATA_DIR="${SMARTMEMORY_DATA_DIR:-$HOME/.smartmemory}"
LOG="$DATA_DIR/plugin.log"
PORT="${SMARTMEMORY_DAEMON_PORT:-9014}"
mkdir -p "$DATA_DIR"
INPUT=$(cat)
LAST_MSG=$(echo "$INPUT" | jq -r '.last_assistant_message // empty')
if [[ -n "$LAST_MSG" ]]; then
    # Try daemon, fall back to CLI — entire block runs in background
    {
        PAYLOAD="{\"content\": $(echo "$LAST_MSG" | jq -Rs .), \"memory_type\": \"episodic\", \"origin\": \"hook:session_end\"}"
        if ! curl -sf -X POST "http://localhost:${PORT}/memory/ingest" \
             -H "Content-Type: application/json" -d "$PAYLOAD" >>"$LOG" 2>&1; then
            python -m smartmemory_app add "$LAST_MSG" >>"$LOG" 2>&1
        fi
    } &
    disown
fi
exit 0
