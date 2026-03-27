#!/usr/bin/env bash
# DIST-AGENT-HOOKS-1: Orient phase — SessionStart hook
# Reads hook JSON from stdin, outputs context to stdout (blocking)
cat | smartmemory lifecycle orient 2>/dev/null
exit 0
