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
    if backend.log_file is not None:
        argv += ["--log-file", str(backend.log_file)]
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


# `-lv 5` emits one of these per layer, e.g.
#   0.00.487.190 D load_tensors: layer   1 assigned to device RPC0, is_swa = 0
# Nothing is printed at default verbosity, so raising it is mandatory for any
# check on where layers actually went.
_LAYER_ASSIGNMENT = re.compile(r"layer\s+(?P<layer>\d+) assigned to device (?P<device>[^\s,]+)")


def parse_layer_assignments(output: str) -> dict[str, list[int]]:
    """Read actual layer placement out of a verbose ``llama-server`` log.

    This is how we verify a plan was honoured rather than assumed: a wrong
    ``--tensor-split`` order does not error, it just quietly runs everything on
    the wrong device.

    The log holds two passes — a memory-fitting dry run and the real load — and
    only the last is what happened. Passes are separated by the layer index
    restarting.
    """
    hits = [
        (int(match.group("layer")), match.group("device"))
        for line in output.splitlines()
        if (match := _LAYER_ASSIGNMENT.search(line))
    ]
    if not hits:
        return {}

    final: list[tuple[int, str]] = []
    previous = -1
    for layer, device in hits:
        if layer <= previous:
            final = []
        final.append((layer, device))
        previous = layer

    placement: dict[str, list[int]] = {}
    for layer, device in final:
        placement.setdefault(device, []).append(layer)
    return placement


# llama.cpp reports e.g. "version: 0.4.1-dev (build 10975, commit 4c9233c03)".
# The build number counts commits since the root, so it depends on clone depth:
# a --depth=1 clone of the *same* commit reports "build 1". Only the commit
# identifies the build for compatibility purposes.
_COMMIT = re.compile(r"commit\s+([0-9a-f]{7,40})")


def version_commit(version: str | None) -> str | None:
    """Extract the llama.cpp commit from a version string."""
    if not version:
        return None
    match = _COMMIT.search(version)
    return match.group(1) if match else None


def same_build(left: str | None, right: str | None) -> bool:
    """Whether two llama.cpp builds can talk to each other over RPC.

    Compares commits, not version strings: the build number varies with clone
    depth even for identical source, and short hashes are truncated to different
    lengths, so one may be a prefix of the other.
    """
    a, b = version_commit(left), version_commit(right)
    if a is None or b is None:
        return True  # unknown on either side: do not block on a guess
    return a.startswith(b) or b.startswith(a)


@dataclass(frozen=True)
class OutOfMemory:
    """What a failed load says about running out of device memory."""

    detected: bool
    # Device ids as the planner names them ("Vulkan0", "RPC0"). Empty when the
    # log shows an OOM but not which device hit it.
    devices: frozenset[str] = frozenset()


# Lines seen on a real Vulkan out-of-memory load failure on omarchy:
#   ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
#   E alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1071374336
#   E llama_model_load: error loading model: unable to allocate Vulkan0 buffer
# "out of memory" covers the CUDA wording. "failed to load model" is deliberately
# not a marker: it also follows failures that have nothing to do with memory.
_OOM_MARKERS = (
    "erroroutofdevicememory",
    "out of memory",
    "unable to allocate",
    "failed to allocate",
)

# Remote devices are printed as "RPC0[100.98.227.49:50052]"; the planner knows
# them as "RPC0", so the bracketed endpoint is dropped.
_OOM_DEVICE = re.compile(
    r"(?:failed|unable) to allocate (?P<device>[A-Za-z_]+\d+)(?:\[[^\]]*\])? buffer"
)


def detect_out_of_memory(lines: list[str]) -> OutOfMemory:
    """Decide whether a failed start ran out of device memory, and where.

    A plan is an estimate. llama.cpp's own --fit only adjusts arguments left
    unset, and we set -ngl and --tensor-split explicitly, so nothing upstream
    checks our numbers before the load fails. This reads the failure afterwards
    so the coordinator can replan instead of giving up.
    """
    detected = False
    devices: set[str] = set()
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in _OOM_MARKERS):
            detected = True
        match = _OOM_DEVICE.search(line)
        if match:
            devices.add(match.group("device"))
    return OutOfMemory(detected=detected, devices=frozenset(devices))


# Output worth keeping from a llama-server run however much else it prints:
# where every layer went, and anything that looks like running out of memory.
DIAGNOSTIC_LINES: tuple[re.Pattern[str], ...] = (
    _LAYER_ASSIGNMENT,
    re.compile("|".join(re.escape(marker) for marker in _OOM_MARKERS), re.IGNORECASE),
)
