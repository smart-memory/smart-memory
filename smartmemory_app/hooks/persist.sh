#!/usr/bin/env bash
# DIST-AGENT-HOOKS-1: Persist phase — SessionEnd hook (async)
# Saves session summary, cleans up state file
{ cat | smartmemory lifecycle persist 2>/dev/null; } &
disown
exit 0
