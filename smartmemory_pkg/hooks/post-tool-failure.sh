#!/usr/bin/env bash
DATA_DIR="${SMARTMEMORY_DATA_DIR:-$HOME/.smartmemory}"
LOG="$DATA_DIR/plugin.log"
mkdir -p "$DATA_DIR"
INPUT=$(cat)
IS_INTERRUPT=$(echo "$INPUT" | jq -r '.is_interrupt // false')
if [[ "$IS_INTERRUPT" == "true" ]]; then exit 0; fi
TOOL=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
ERROR=$(echo "$INPUT" | jq -r '.error // empty')
if [[ -n "$ERROR" ]]; then
    CONTENT="Tool $TOOL failed: $ERROR"
    python -m smartmemory_pkg ingest "$CONTENT" --type episodic 2>>"$LOG"
fi
exit 0
