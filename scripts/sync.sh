#!/usr/bin/env bash
# Push this working tree to a target node.
#
# The dev box is the only source of truth for code; nothing is edited in place
# on a target. CLAUDE.md is deliberately untracked by git but IS synced here —
# rsync does not read .gitignore, and it is how a session on the target box
# gets project context.
#
# Usage:
#   scripts/sync.sh edgeup@192.168.0.54
#   HUDDLE_TARGET=edgeup@192.168.0.54 scripts/sync.sh
set -euo pipefail

TARGET="${1:-${HUDDLE_TARGET:-}}"
DEST="${HUDDLE_REMOTE_DIR:-~/huddle}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$TARGET" ]; then
    echo "usage: $0 <user@host>   (or set HUDDLE_TARGET)" >&2
    exit 2
fi

# huddle.yaml is node-local: it exists only on the target and names that box's
# binary paths. --delete would otherwise wipe it on every sync.
exec rsync -az --delete --info=stats1 \
    --exclude 'huddle.yaml' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '*.gguf' \
    --exclude 'models/' \
    --exclude 'llama.cpp/' \
    "$REPO_ROOT/" "$TARGET:$DEST/"
