# Huddle

[![ci](https://github.com/sszxt/huddle/actions/workflows/ci.yml/badge.svg)](https://github.com/sszxt/huddle/actions/workflows/ci.yml)

Run GGUF models across several Linux machines when no single box has enough
memory to hold them.

exo does this today, but GPU-accelerates only on macOS — its engine is MLX, so
Linux falls back to CPU. Huddle fills that gap by orchestrating llama.cpp's RPC
backend instead of porting MLX: cross-vendor by way of Vulkan, so NVIDIA, AMD,
Intel and CPU-only nodes can share one cluster.

Huddle does not implement inference or model parallelism. llama.cpp already
splits layers across networked `rpc-server` processes. Huddle decides the split,
starts and supervises the processes, and puts an OpenAI-compatible API in front.

## Status

Two-node cluster (two RTX 5070s, one Arch and one Ubuntu node): a 32B Q4_K_M
model that fits on neither GPU alone serves at 27.7 tok/s generation, fully
offloaded and split across both.

- **v0** — single node behind an OpenAI-compatible API
- **v1** — multi-node layer split, supervision and restart, model switching,
  systemd units
- **v2** — peer discovery over mDNS
- **v3** — capacity-aware placement (out-of-memory replanning, peer rejoin)
  plus `huddle tui`, a terminal dashboard for watching and controlling the
  cluster, with best-effort GPU utilization/temperature/power and generation
  speed. The dashboard and its metrics are demoed and unit-tested against fake
  binaries; not yet confirmed against a live cluster.

More nodes add capacity, not speed: pipeline parallelism runs one stage at a
time, and every node boundary costs a network round trip per token.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- llama.cpp built with `-DGGML_RPC=ON`, and `-DGGML_VULKAN=1` for GPU offload

## Quick start

```sh
uv sync
cp huddle.example.yaml huddle.yaml   # point it at your llama.cpp build and models
uv run huddle doctor                 # checks the node, peers and live cluster
uv run huddle plan                   # shows how layers would be split
uv run huddle serve                  # agent + OpenAI-compatible API
uv run huddle tui                    # terminal dashboard: monitor and control it
```

On worker nodes, `uv run huddle agent` runs the agent alone.
`scripts/bootstrap-node.sh` builds llama.cpp at the commit pinned in
`llamacpp.pin` — every node must run the same build — and
`scripts/install-service.sh` installs a systemd unit.

`scripts/gpu-watch.py` shows GPU memory held by llama.cpp on each node,
separately from desktop applications.

## Security

The llama.cpp RPC backend has no authentication and no encryption, and upstream
states it must never run on an open network. Bind `rpc-server` to a private LAN
or WireGuard interface and firewall the port. Running the cluster over Tailscale
covers both. The node agent's control API has no authentication either; keep
it off public interfaces.

## Tests

The test suite runs against fake llama.cpp binaries in `tests/fakes/`, which
record the command lines Huddle builds and reproduce llama.cpp's layer placement.
A green run shows the code does what it was told; only a real cluster shows it
works.
