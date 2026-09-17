#!/usr/bin/env bash
# Install Huddle as a systemd service on this node.
#
# The unit is generated from the local environment rather than copied from a
# template, because user, home directory and uv location differ per node — the
# checked-in units under packaging/ are examples, not something to deploy as-is.
#
# Usage, run ON the node:
#   scripts/install-service.sh              # full node: agent + API + cluster
#   scripts/install-service.sh --agent      # worker node: agent only
set -euo pipefail

MODE=full
[ "${1:-}" = "--agent" ] && MODE=agent

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Standard sudo flushes typed-ahead input before prompting, so a piped password
# is discarded even over a pty. SUDO_ASKPASS is the supported way to install
# unattended; without it this simply prompts as usual.
SUDO=(sudo)
[ -n "${SUDO_ASKPASS:-}" ] && SUDO=(sudo -A)
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || { echo "error: uv not found" >&2; exit 1; }

# Run the environment's own entry point rather than `uv run`. Under `uv run` the
# unit's main process is uv, which exits 143 when systemd stops it — so every
# clean stop or restart was recorded as a failure — and killing the main PID
# left the real server running for a moment. Deployments run `uv sync` first.
(cd "$REPO_ROOT" && "$UV" sync --locked --quiet)
HUDDLE="$REPO_ROOT/.venv/bin/huddle"
[ -x "$HUDDLE" ] || { echo "error: $HUDDLE not found after uv sync" >&2; exit 1; }

if [ "$MODE" = agent ]; then
    UNIT=huddle-agent.service
    DESC="Huddle node agent (worker)"
    CMD="$HUDDLE agent"
else
    UNIT=huddle.service
    DESC="Huddle node (agent + OpenAI-compatible API)"
    CMD="$HUDDLE serve"
fi

echo "==> installing $UNIT for $USER in $REPO_ROOT"

"${SUDO[@]}" tee "/etc/systemd/system/$UNIT" >/dev/null <<UNITFILE
[Unit]
Description=$DESC
Documentation=https://github.com/sszxt/huddle
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=$USER
WorkingDirectory=$REPO_ROOT
Environment=PATH=$REPO_ROOT/.venv/bin:$(dirname "$UV"):/usr/local/bin:/usr/bin:/bin
ExecStart=$CMD
Restart=on-failure
RestartSec=5
# 143 is SIGTERM's exit status: a stop we asked for, not a failure.
SuccessExitStatus=143

# A large GGUF takes real time to load, and a head node streams every remote
# layer's weights to its peers before it will answer. Do not kill a slow start.
TimeoutStartSec=900

# Reach the whole tree: llama.cpp runs as a child, and signalling only the
# parent leaves it holding its port and its GPU memory.
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
UNITFILE

"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable "$UNIT"
echo "==> installed and enabled. Start it with: sudo systemctl start $UNIT"
