#!/usr/bin/env bash
# DIST-AGENT-HOOKS-1: Learn phase — PostToolUseFailure hook (async)
{ cat | smartmemory lifecycle learn 2>/dev/null; } &
disown
exit 0
