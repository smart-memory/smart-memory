#!/usr/bin/env bash
# DIST-DAEMON-1: Try daemon (instant), fall back to CLI (slow)
DATA_DIR="${SMARTMEMORY_DATA_DIR:-$HOME/.smartmemory}"
LOG="$DATA_DIR/plugin.log"
PORT="${SMARTMEMORY_DAEMON_PORT:-9014}"
mkdir -p "$DATA_DIR"
INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
RESULT=$(curl -sf --get "http://localhost:${PORT}/memory/recall" \
    --data-urlencode "cwd=$CWD" --data-urlencode "top_k=10" 2>>"$LOG")
if [[ -n "$RESULT" ]]; then
    echo "$RESULT" | jq -r '.context // empty'
else
    python -m smartmemory_app recall ${CWD:+--cwd "$CWD"} 2>>"$LOG"
fi
exit 0
