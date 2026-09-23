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

Two-node cluster, both nodes running Ubuntu (two RTX 5070s): a 32B Q4_K_M
model that fits on neither GPU alone serves fully offloaded and split across
both. Previously benchmarked at 27.7 tok/s generation at full 64/64 layer
offload; the head node has since had its OS reinstalled, and `huddle doctor`
confirms the rebuilt cluster is back up and placing layers correctly
(63/64 offloaded at last check, the remainder depending on free VRAM at
load time). The two nodes were originally different distros (Arch and
Ubuntu); after the reinstall both run Ubuntu, so that cross-distro case is
no longer exercised here.

- **v0** — single node behind an OpenAI-compatible API
- **v1** — multi-node layer split, supervision and restart, model switching,
  systemd units
- **v2** — peer discovery over mDNS
- **v3** — capacity-aware placement (out-of-memory replanning, peer rejoin)
  plus `huddle tui`, a terminal dashboard built with `rich` (modeled on exo's
  own topology view) for watching and controlling the cluster, with GPU
  utilization/temperature/power and generation speed. Demoed against a live
  cluster on real hardware, including real GPU metrics and measured tok/s.
- **Model downloads** — search Hugging Face and pull a GGUF straight to a
  node's model directory, no manual `scp`/`hf download` required. Available
  both from `huddle tui` and from the web UI's Workspace page. Unit-tested
  against fakes and demoed manually; not yet exercised end-to-end against the
  real Hugging Face Hub through either UI.
- **Web UI** — a chat front end at `/ui`, laid out after Open WebUI's chat
  screen: streaming chat against the cluster's OpenAI-compatible API, model
  switching, chat history kept in the browser, and the model manager under
  Workspace. Plain HTML, CSS and JS, no build step. Checked in a browser
  against the fake binaries only; not yet run against a live cluster.

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
uv run huddle serve                  # agent + OpenAI-compatible API + web UI at /ui
uv run huddle tui                    # terminal dashboard: monitor, control, download models
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
or VPN interface and firewall the port so only cluster nodes can reach it — a
host firewall like `ufw` allowing the RPC port from peer addresses only is
enough on a trusted LAN; a WireGuard/Tailscale overlay adds encryption on top
if the network isn't otherwise trusted. The node agent's control API and the
web UI (`/ui`, which drives the `/cluster/*` routes to switch and download
models) have no authentication either; keep all of it off public interfaces.
Only chat requests (`/v1/*`) check `api.api_key`, and the web UI asks for the
key when one is set.

## Tests

The test suite runs against fake llama.cpp binaries in `tests/fakes/`, which
record the command lines Huddle builds and reproduce llama.cpp's layer placement.
A green run shows the code does what it was told; only a real cluster shows it
works.
