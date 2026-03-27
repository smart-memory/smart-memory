#!/usr/bin/env bash
# DIST-AGENT-HOOKS-1: Recall phase — UserPromptSubmit hook
# Always captures prompt for distill pairing; optionally injects context (blocking)
cat | smartmemory lifecycle recall 2>/dev/null
exit 0
