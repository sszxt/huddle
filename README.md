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

## Install

Run this on every Linux PC that should share its GPU:

```sh
curl -fsSL https://raw.githubusercontent.com/sszxt/huddle/main/install.sh | sh
```

Then open <http://localhost:8000> on any of them. The PCs find each other on
the local network; there is nothing to configure. The home screen shows how
much the cluster can hold and offers models that fit — one click downloads a
model and starts it, borrowing the other PCs' GPUs when it needs them.

The installer asks for your password once (sudo) and:

- installs Huddle with [uv](https://docs.astral.sh/uv/), which brings its own
  Python;
- downloads llama.cpp's official prebuilt Vulkan release at the pinned build
  and checks its checksum — nothing is compiled; only the Vulkan runtime and
  your GPU driver are needed;
- starts Huddle as a systemd service, so it is back after every reboot;
- opens Huddle's two ports (8000, 50052) to the local subnet only, if `ufw`
  or `firewalld` is filtering.

Re-run the same command to upgrade. `huddle uninstall` removes the service,
firewall rules and llama.cpp (`--purge` also deletes downloaded models).
Pass options through with `sh -s --`, e.g. `| sh -s -- --cluster lab` to keep
this group of PCs separate from others on the same network.

### How the PCs work together

Every PC runs the same thing. The one where you start a model becomes the
*head*: it holds the model file and borrows GPUs from the others, which each
run a worker for it. You can chat from any PC's page — requests go to the
head — and starting a model on another PC hands the cluster over to it. One
model runs at a time. A PC that joins later is used the next time a model is
loaded, or straight away if the running model has layers on CPU it could take.
The head brings its model back after a reboot, waiting briefly for the PCs it
last ran with.

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
- **Install and automatic clustering** — the one-line installer above, zero
  configuration, every PC equal (whichever one loads a model is the head),
  mDNS membership with worker leases, handover between PCs, and a first-run
  screen that offers models sized to the cluster. The prebuilt llama.cpp
  release was checked against the source build on the real head node
  (identical devices, placement and speed); the rest is tested against the
  fakes, including two nodes in one process on an in-memory network. Not
  yet run as a two-PC install on the real hardware.
- **Web UI** — a chat front end at `/ui`, laid out after Open WebUI's chat
  screen: streaming chat against the cluster's OpenAI-compatible API, model
  switching, chat history kept in the browser, and the model manager under
  Workspace. A Cluster page maps every node live: addresses, CPU, memory, each
  GPU's VRAM/utilization/temperature/power, round-trip time from the head, and
  which layers sit on which device. Plain HTML, CSS and JS, no build step.
  Checked in a browser against the fake binaries only; not yet run against a
  live cluster.

More nodes add capacity, not speed: pipeline parallelism runs one stage at a
time, and every node boundary costs a network round trip per token.

## Requirements

- Linux, x86_64 or arm64, with systemd. The prebuilt llama.cpp is built on
  Ubuntu; other glibc distributions usually work but are not tested here.
- A GPU with a Vulkan driver (NVIDIA, AMD or Intel). Without one, models run
  on CPU.
- PCs on the same local network, where multicast (mDNS) reaches between them.

## Commands

```sh
huddle doctor      # checks this PC, its peers and the live cluster
huddle discover    # lists the Huddle PCs on this network
huddle plan        # shows how a model's layers would be split
huddle tui         # terminal dashboard: monitor, control, download models
huddle setup       # re-run the install steps (what install.sh ends with)
```

Configuration is optional. Settings go in `~/.config/huddle/huddle.yaml`
(`huddle.example.yaml` lists every key); a `huddle.yaml` in the working
directory or `$HUDDLE_CONFIG` takes precedence, which is how a checkout runs.

## Development

```sh
uv sync
uv run huddle serve                  # from a checkout, with ./huddle.yaml or no config
uv run pytest && uv run ruff check && uv run mypy
```

`HUDDLE_SOURCE=path/to/checkout sh install.sh` installs a working tree instead
of the GitHub release. `scripts/bootstrap-node.sh` builds llama.cpp from source
at the pinned commit (`src/huddle/llamacpp.pin`) for what the prebuilt release
does not cover, such as CUDA; every node must run the same build.

`scripts/gpu-watch.py` shows GPU memory held by llama.cpp on each node,
separately from desktop applications.

## Security

**Huddle trusts your local network.** PCs join each other automatically, with
no pairing, so anything on the same network can use the cluster: the web UI and
API on port 8000 (also reachable from other devices at `http://<pc>:8000`), the
node control routes, and llama.cpp's RPC backend on port 50052, which has no
authentication and no encryption — upstream states it must never run on an
open network. Use Huddle on a home or lab network you control, never on public
or guest Wi-Fi. The installer opens the ports to the local subnet only; bind
`api.host: 127.0.0.1` to keep a PC from lending its GPU or serving its UI to
others. A WireGuard/Tailscale overlay (`discovery.advertise`) adds encryption
if the network is not otherwise trusted. Only chat requests (`/v1/*`) check
`api.api_key`, which must then match on every PC; the web UI asks for it.

## Tests

The test suite runs against fake llama.cpp binaries in `tests/fakes/`, which
record the command lines Huddle builds and reproduce llama.cpp's layer placement.
A green run shows the code does what it was told; only a real cluster shows it
works.
