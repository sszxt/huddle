#!/usr/bin/env bash
# Build llama.cpp for a Huddle node.
#
# Every node must build from the pinned commit in llamacpp.pin: llama.cpp's RPC
# protocol performs a version handshake and rejects mismatched peers.
#
# Usage:
#   scripts/bootstrap-node.sh              # Vulkan build (default)
#   HUDDLE_BACKEND=cpu scripts/bootstrap-node.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../llamacpp.pin
source "$REPO_ROOT/llamacpp.pin"

SRC_DIR="${HUDDLE_LLAMACPP_DIR:-$HOME/llama.cpp}"
BACKEND="${HUDDLE_BACKEND:-vulkan}"
JOBS="${HUDDLE_JOBS:-$(nproc)}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

install_deps() {
    local common_arch=(cmake ninja ccache git)
    local common_deb=(cmake ninja-build ccache git build-essential)

    if command -v pacman >/dev/null; then
        local pkgs=("${common_arch[@]}")
        [ "$BACKEND" = vulkan ] && pkgs+=(vulkan-headers spirv-headers shaderc vulkan-icd-loader)
        log "installing via pacman: ${pkgs[*]}"
        sudo pacman -S --needed --noconfirm "${pkgs[@]}"
    elif command -v apt-get >/dev/null; then
        local pkgs=("${common_deb[@]}")
        [ "$BACKEND" = vulkan ] && pkgs+=(libvulkan-dev glslc spirv-headers)
        log "installing via apt: ${pkgs[*]}"
        sudo apt-get update && sudo apt-get install -y "${pkgs[@]}"
    else
        die "unsupported package manager; install cmake, ninja and the Vulkan SDK manually"
    fi
}

fetch_source() {
    if [ -d "$SRC_DIR/.git" ]; then
        log "updating $SRC_DIR"
        git -C "$SRC_DIR" fetch origin "$LLAMACPP_REF"
    else
        log "cloning llama.cpp into $SRC_DIR"
        git clone "$LLAMACPP_REPO" "$SRC_DIR"
    fi
    git -C "$SRC_DIR" checkout -f "$LLAMACPP_REF"
    log "pinned at $(git -C "$SRC_DIR" rev-parse --short HEAD)"
}

build() {
    local flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON)
    if [ "$BACKEND" = vulkan ]; then
        flags+=(-DGGML_VULKAN=ON)
    else
        log "building CPU-only (HUDDLE_BACKEND=$BACKEND)"
    fi
    command -v ccache >/dev/null && flags+=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    )

    log "configuring"
    cmake -B "$SRC_DIR/build" -S "$SRC_DIR" -G Ninja "${flags[@]}"
    log "building with $JOBS jobs"
    cmake --build "$SRC_DIR/build" --config Release -j "$JOBS"
}

verify() {
    local bin="$SRC_DIR/build/bin"
    [ -x "$bin/llama-server" ] || die "llama-server was not built"

    # Upstream's README says "rpc-server" but the build produces "ggml-rpc-server".
    # Accept either, so this keeps working whichever name a given commit uses.
    local rpc_bin=""
    for candidate in ggml-rpc-server rpc-server; do
        if [ -x "$bin/$candidate" ]; then rpc_bin="$bin/$candidate"; break; fi
    done
    [ -n "$rpc_bin" ] || die "no RPC server binary built — is -DGGML_RPC=ON set?"
    log "built: $("$bin/llama-server" --version 2>&1 | head -1)"
    "$bin/llama-server" --list-devices || true
    cat <<MSG

Binaries are in $bin
Point huddle.yaml at them:

  binaries:
    llama_server: $bin/llama-server
    rpc_server:   $rpc_bin
MSG
}

install_deps
fetch_source
build
verify
