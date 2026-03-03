#!/usr/bin/env bash
DATA_DIR="${SMARTMEMORY_DATA_DIR:-$HOME/.smartmemory}"
LOG="$DATA_DIR/plugin.log"
mkdir -p "$DATA_DIR"
INPUT=$(cat)
LAST_MSG=$(echo "$INPUT" | jq -r '.last_assistant_message // empty')
if [[ -n "$LAST_MSG" ]]; then
    python -m smartmemory_pkg persist "$LAST_MSG" 2>>"$LOG"
fi
exit 0
