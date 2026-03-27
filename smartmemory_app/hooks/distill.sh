#!/usr/bin/env bash
# DIST-AGENT-HOOKS-1: Distill phase — Stop hook (async)
# Pairs last_assistant_message with stored prompt
{ cat | smartmemory lifecycle distill 2>/dev/null; } &
disown
exit 0
