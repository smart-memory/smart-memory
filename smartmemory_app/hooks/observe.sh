#!/usr/bin/env bash
# DIST-AGENT-HOOKS-1: Observe phase — PostToolUse hook (async)
{ cat | smartmemory lifecycle observe 2>/dev/null; } &
disown
exit 0
