"""Best-effort GPU utilization, temperature and power, via ``nvidia-smi``.

Device placement never depends on this (``hardware.py`` deliberately reads
llama.cpp's own ``--list-devices`` for that), but a live dashboard wants to
show more than free/total memory. This is NVIDIA-only and always optional:
AMD/Intel devices, and any node without ``nvidia-smi`` on PATH, simply report
nothing here rather than failing.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

_QUERY = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"


@dataclass(frozen=True)
class GpuSample:
    """One GPU as ``nvidia-smi`` reports it right now."""

    index: int
    name: str
    used_mib: int
    total_mib: int
    util_pct: int
    temp_c: int
    power_w: float | None


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _parse_power(value: str) -> float | None:
    # Unsupported cards/drivers report "[N/A]" for power.draw.
    try:
        return float(value)
    except ValueError:
        return None


def parse_nvidia_smi_csv(output: str) -> list[GpuSample]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output.

    Deliberately tolerant: a line that does not parse is skipped rather than
    raising, since this is informational and must never break the dashboard.
    """
    samples: list[GpuSample] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 7:
            continue
        index = _parse_int(fields[0])
        used = _parse_int(fields[2])
        total = _parse_int(fields[3])
        util = _parse_int(fields[4])
        temp = _parse_int(fields[5])
        if index is None or used is None or total is None or util is None or temp is None:
            continue
        samples.append(
            GpuSample(
                index=index,
                name=fields[1],
                used_mib=used,
                total_mib=total,
                util_pct=util,
                temp_c=temp,
                power_w=_parse_power(fields[6]),
            )
        )
    return samples


def probe_gpu_samples(timeout: float = 5.0) -> list[GpuSample]:
    """Ask ``nvidia-smi`` for live GPU stats. Never raises; ``[]`` on failure.

    A node with no NVIDIA GPU, no driver, or a busy/misbehaving ``nvidia-smi``
    must never take the rest of the hardware probe down with it.
    """
    if not nvidia_smi_available():
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_nvidia_smi_csv(result.stdout)


__all__ = ["GpuSample", "nvidia_smi_available", "parse_nvidia_smi_csv", "probe_gpu_samples"]
