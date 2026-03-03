#!/usr/bin/env bash
DATA_DIR="${SMARTMEMORY_DATA_DIR:-$HOME/.smartmemory}"
LOG="$DATA_DIR/plugin.log"
mkdir -p "$DATA_DIR"
INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
python -m smartmemory_pkg recall ${CWD:+--cwd "$CWD"} 2>>"$LOG"
exit 0
