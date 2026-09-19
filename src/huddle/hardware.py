"""What this node can contribute to a cluster.

Device information comes from llama.cpp itself rather than vulkaninfo or
nvidia-smi: what matters for placement is what the inference engine can
actually see and use, and those sources disagree often enough to matter.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from huddle.gpu import GpuSample, probe_gpu_samples
from huddle.llamacpp import Device, LlamaCppError, binary_version, list_devices

MEMINFO = Path("/proc/meminfo")


class DeviceInfo(BaseModel):
    """A compute device as llama.cpp reports it."""

    id: str
    name: str
    total_mib: int | None = None
    free_mib: int | None = None
    is_rpc: bool = False
    unified_memory: bool = False
    # Best-effort, NVIDIA-only, from `nvidia-smi` — never used for placement,
    # only for display. None on non-NVIDIA devices or without nvidia-smi.
    util_pct: int | None = None
    temp_c: int | None = None
    power_w: float | None = None

    @classmethod
    def from_device(cls, device: Device) -> DeviceInfo:
        return cls(
            id=device.id,
            name=device.name,
            total_mib=device.total_mib,
            free_mib=device.free_mib,
            is_rpc=device.is_rpc,
            unified_memory=_looks_unified(device),
        )

    @property
    def usable_mib(self) -> int | None:
        """Memory we are willing to place layers in.

        A unified-memory device reports system RAM as if it were VRAM — on our
        own hardware an Intel iGPU advertises 48 GiB next to a 12 GiB discrete
        card. Treating that as dedicated would hand it most of the model, so
        placement must discount it rather than trust the number.
        """
        if self.total_mib is None:
            return None
        if self.unified_memory:
            return None
        return self.free_mib if self.free_mib is not None else self.total_mib


class NodeHardware(BaseModel):
    """Everything the coordinator needs to place layers on this node."""

    name: str
    llamacpp_version: str | None = None
    devices: list[DeviceInfo] = []
    cpu_count: int = 0
    ram_total_mib: int = 0
    ram_available_mib: int = 0
    error: str | None = None

    @property
    def gpu_usable_mib(self) -> int:
        """Total memory on devices we would actually place layers on."""
        return sum(device.usable_mib or 0 for device in self.devices if not device.is_rpc)


# Vendor-neutral markers for integrated GPUs, which share system RAM. llama.cpp
# prints a `uma: 1` flag on its Vulkan init lines, but not in --list-devices, so
# we fall back to naming until the agent parses backend startup output.
_UNIFIED_HINTS = ("(rpl-", "(adl-", "(tgl-", "uhd graphics", "iris", "radeon graphics", "vega")


def _looks_unified(device: Device) -> bool:
    lowered = device.name.lower()
    return any(hint in lowered for hint in _UNIFIED_HINTS)


_NVIDIA_HINTS = ("nvidia", "geforce", "rtx", "quadro", "tesla")


def _looks_nvidia(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _NVIDIA_HINTS)


def _same_gpu(device_name: str, sample_name: str) -> bool:
    a, b = device_name.lower(), sample_name.lower()
    return a in b or b in a


def _with_gpu_samples(devices: list[DeviceInfo], samples: list[GpuSample]) -> list[DeviceInfo]:
    """Best-effort match of `nvidia-smi` samples onto llama.cpp's device list.

    llama.cpp's device order and nvidia-smi's device order are not guaranteed
    to agree, so this matches by name first. A single candidate device left
    over alongside a single leftover sample is paired positionally; anything
    still ambiguous (e.g. two unnamed matches) is left unmatched rather than
    guessed, since a wrong pairing would be worse than a missing one.
    """
    if not samples:
        return devices

    remaining = list(samples)
    matched: dict[int, GpuSample] = {}
    for i, device in enumerate(devices):
        if not _looks_nvidia(device.name):
            continue
        hit = next((s for s in remaining if _same_gpu(device.name, s.name)), None)
        if hit is not None:
            matched[i] = hit
            remaining.remove(hit)

    unmatched_indices = [
        i for i, d in enumerate(devices) if _looks_nvidia(d.name) and i not in matched
    ]
    if len(unmatched_indices) == 1 and len(remaining) == 1:
        matched[unmatched_indices[0]] = remaining[0]

    if not matched:
        return devices
    return [
        (
            device.model_copy(
                update={
                    "util_pct": matched[i].util_pct,
                    "temp_c": matched[i].temp_c,
                    "power_w": matched[i].power_w,
                }
            )
            if i in matched
            else device
        )
        for i, device in enumerate(devices)
    ]


def read_memory() -> tuple[int, int]:
    """Return ``(total_mib, available_mib)`` for system RAM."""
    if not MEMINFO.exists():
        return (0, 0)
    values: dict[str, int] = {}
    for line in MEMINFO.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(rest.strip().split()[0]) // 1024
    return (values.get("MemTotal", 0), values.get("MemAvailable", 0))


def probe(name: str, llama_server: Path) -> NodeHardware:
    """Describe this node.

    Never raises: a node that cannot report its devices should still appear in
    the cluster with an explanatory error, because a missing node is far harder
    to debug than one reporting why it is unusable.
    """
    total, available = read_memory()
    hardware = NodeHardware(
        name=name,
        cpu_count=os.cpu_count() or 0,
        ram_total_mib=total,
        ram_available_mib=available,
    )
    try:
        devices = [DeviceInfo.from_device(d) for d in list_devices(llama_server)]
        hardware.devices = _with_gpu_samples(devices, probe_gpu_samples())
        hardware.llamacpp_version = binary_version(llama_server)
    except LlamaCppError as exc:
        hardware.error = str(exc)
    return hardware
