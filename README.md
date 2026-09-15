# Huddle

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

Early. v0 (single machine, OpenAI-compatible API) is under construction.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- llama.cpp built with `-DGGML_RPC=ON`, and `-DGGML_VULKAN=1` for GPU offload

## Quick start

```sh
uv sync
uv run huddle --help
```

## Security

The llama.cpp RPC backend has no authentication and no encryption, and upstream
states it must never run on an open network. Bind `rpc-server` to a private LAN
or WireGuard interface and firewall the port.
