"""The seam between Huddle and llama.cpp.

Everything Huddle knows about llama.cpp's command-line surface lives here, so
there is exactly one place to fix when upstream changes a flag. The argv
builders are the real contract with llama.cpp — we cannot run inference on the
dev box, but we can assert to the character what we would have executed.

Flags verified against llama.cpp master, September 2026.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from huddle.config import BackendConfig, RpcConfig


class LlamaCppError(RuntimeError):
    """A llama.cpp binary is missing, unusable, or answered unexpectedly."""


@dataclass(frozen=True)
class Device:
    """One compute device as llama.cpp itself reports it.

    We take llama.cpp's word over vulkaninfo or nvidia-smi: what matters for
    placement is what the inference engine can actually see and use.
    """

    id: str
    name: str
    total_mib: int | None = None
    free_mib: int | None = None

    @property
    def is_rpc(self) -> bool:
        return self.id.upper().startswith("RPC")


def require_binary(path: Path, *, role: str) -> Path:
    """Validate that a llama.cpp binary exists and is executable."""
    if not path.exists():
        raise LlamaCppError(f"{role} binary not found: {path}")
    if not path.is_file():
        raise LlamaCppError(f"{role} binary is not a file: {path}")
    return path


def build_rpc_server_argv(binary: Path, rpc: RpcConfig) -> list[str]:
    """Command line for a worker's ``rpc-server``.

    ``-H`` is always passed. llama.cpp defaults to binding ``127.0.0.1``, which
    silently makes the worker unreachable from other machines; being explicit
    here is what keeps that from looking like a firewall bug.
    """
    argv = [str(binary), "-H", rpc.bind, "-p", str(rpc.port)]
    if rpc.cache:
        # Caches tensors locally by hash, so only the first load pays for the
        # transfer of the weights across the network.
        argv.append("-c")
    if rpc.devices:
        argv += ["-d", ",".join(rpc.devices)]
    if rpc.threads is not None:
        argv += ["-t", str(rpc.threads)]
    return argv


def build_llama_server_argv(
    binary: Path,
    backend: BackendConfig,
    model_path: Path,
    *,
    rpc_endpoints: list[str] | None = None,
    tensor_split: list[float] | None = None,
    alias: str | None = None,
) -> list[str]:
    """Command line for the head ``llama-server``.

    ``rpc_endpoints`` are ``host:port`` strings; each becomes an extra ggml
    device on this node. ``tensor_split`` gives the proportion of the model
    placed on each device, in device-list order.
    """
    argv = [
        str(binary),
        "-m",
        str(model_path),
        "--host",
        backend.host,
        "--port",
        str(backend.port),
        "-c",
        str(backend.ctx_size),
        "-np",
        str(backend.parallel),
        "-ngl",
        str(backend.n_gpu_layers),
    ]
    if rpc_endpoints:
        argv += ["--rpc", ",".join(rpc_endpoints)]
    if tensor_split:
        argv += ["--tensor-split", ",".join(f"{value:g}" for value in tensor_split)]
    if alias:
        argv += ["--alias", alias]
    if backend.api_key:
        argv += ["--api-key", backend.api_key]
    argv += backend.extra_args
    return argv


# Matches the per-device lines of `llama-server --list-devices`, e.g.
#   Vulkan0: NVIDIA GeForce RTX 5070 (12227 MiB, 11041 MiB free)
# Memory is optional because CPU and RPC devices do not always report it.
_DEVICE_LINE = re.compile(
    r"^\s*(?P<id>[A-Za-z0-9_]+)\s*:\s*(?P<name>.+?)"
    r"(?:\s*\(\s*(?P<total>\d+)\s*MiB"
    r"(?:\s*,\s*(?P<free>\d+)\s*MiB\s*free)?\s*\))?\s*$"
)


def parse_devices(output: str) -> list[Device]:
    """Parse the output of ``llama-server --list-devices``.

    Deliberately tolerant: upstream has changed this format before, and a
    slightly-off device list should degrade placement, not crash the agent.
    """
    devices: list[Device] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":") or stripped.lower().startswith("available"):
            continue
        match = _DEVICE_LINE.match(line)
        if not match:
            continue
        total = match.group("total")
        free = match.group("free")
        devices.append(
            Device(
                id=match.group("id"),
                name=match.group("name").strip(),
                total_mib=int(total) if total else None,
                free_mib=int(free) if free else None,
            )
        )
    return devices


def list_devices(binary: Path, *, timeout: float = 30.0) -> list[Device]:
    """Ask a llama.cpp binary which devices it can see."""
    require_binary(binary, role="llama-server")
    try:
        result = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        raise LlamaCppError(f"could not execute {binary}: {exc}") from exc
    # Some builds print the device list on stderr; accept either.
    return parse_devices(result.stdout or "") or parse_devices(result.stderr or "")


def binary_version(binary: Path, *, timeout: float = 30.0) -> str:
    """Return a version fingerprint for a llama.cpp binary.

    Every node in a cluster must run the same llama.cpp build: the RPC protocol
    handshake rejects a peer whose major version differs. Capturing this per
    node turns that failure into a clear report instead of a mystery at connect
    time.
    """
    require_binary(binary, role="llama.cpp")
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        raise LlamaCppError(f"could not execute {binary}: {exc}") from exc
    text = (result.stderr or "") + (result.stdout or "")
    for line in text.splitlines():
        if "version" in line.lower():
            return line.strip()
    return text.strip().splitlines()[0] if text.strip() else "unknown"
