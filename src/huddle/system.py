"""What a node's operating system says about it, for the web UI's cluster page.

Display only, like the ``nvidia-smi`` figures in ``gpu.py``: placement never
reads any of this. Standard library only, from Linux's ``/proc`` and ``/sys``
plus iproute2's ``ip``, which every target node has. Each parser takes text and
is pure, so it is testable from fixtures; ``probe_system`` does the I/O and
never raises — a missing source leaves its field empty rather than hiding the
whole node.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel

PROC = Path("/proc")
OS_RELEASE = Path("/etc/os-release")
SYS_NET = Path("/sys/class/net")


class NetAddress(BaseModel):
    """One IPv4 address on one interface."""

    interface: str
    address: str
    prefix: int | None = None
    # Negotiated link speed, when the kernel reports one (wired NICs do).
    link_mbps: int | None = None


class SystemInfo(BaseModel):
    """The node as its operating system describes it."""

    hostname: str
    os: str | None = None
    kernel: str | None = None
    cpu_model: str | None = None
    cpu_threads: int = 0
    # Busy share of all CPUs over a short sample, 0-100.
    cpu_pct: float | None = None
    load_1m: float | None = None
    uptime_s: float | None = None
    addresses: list[NetAddress] = []


def parse_os_release(text: str) -> str | None:
    """``PRETTY_NAME`` from ``/etc/os-release``, e.g. "Ubuntu 26.04.1 LTS"."""
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "PRETTY_NAME":
            return value.strip().strip('"').strip("'") or None
    return None


def parse_cpu_model(text: str) -> str | None:
    """The first ``model name`` in ``/proc/cpuinfo``."""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return " ".join(value.split()) or None
    return None


def parse_cpu_times(text: str) -> tuple[int, int] | None:
    """``(idle, total)`` jiffies from the aggregate ``cpu`` line of ``/proc/stat``.

    Idle includes iowait. Only the first eight fields count: guest time is
    already included in user time, so adding it would count it twice.
    """
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "cpu":
            continue
        try:
            values = [int(v) for v in fields[1:9]]
        except ValueError:
            return None
        if len(values) < 4:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return idle, sum(values)
    return None


def cpu_percent(before: tuple[int, int], after: tuple[int, int]) -> float | None:
    """Busy share of the time between two ``parse_cpu_times`` samples."""
    idle = after[0] - before[0]
    total = after[1] - before[1]
    if total <= 0:
        return None
    return round(100.0 * (total - idle) / total, 1)


def parse_uptime(text: str) -> float | None:
    """Seconds since boot, the first field of ``/proc/uptime``."""
    try:
        return float(text.split()[0])
    except (IndexError, ValueError):
        return None


def parse_loadavg(text: str) -> float | None:
    """The one-minute load average, the first field of ``/proc/loadavg``."""
    try:
        return float(text.split()[0])
    except (IndexError, ValueError):
        return None


def parse_ip_addresses(output: str) -> list[NetAddress]:
    """IPv4 addresses from ``ip -j -4 addr show``, loopback left out."""
    try:
        interfaces = json.loads(output)
    except json.JSONDecodeError:
        return []
    if not isinstance(interfaces, list):
        return []
    addresses: list[NetAddress] = []
    for interface in interfaces:
        if not isinstance(interface, dict):
            continue
        name = interface.get("ifname")
        if not name or "LOOPBACK" in interface.get("flags", []):
            continue
        for info in interface.get("addr_info", []):
            if info.get("family") == "inet" and info.get("local"):
                addresses.append(
                    NetAddress(interface=name, address=info["local"], prefix=info.get("prefixlen"))
                )
    return addresses


def link_speed(interface: str, sys_net: Path = SYS_NET) -> int | None:
    """Link speed in Mb/s. Wireless, virtual and down links report none."""
    text = _read(sys_net / interface / "speed")
    if text is None:
        return None
    try:
        speed = int(text.strip())
    except ValueError:
        return None
    return speed if speed > 0 else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _addresses(sys_net: Path) -> list[NetAddress]:
    ip = shutil.which("ip")
    if ip is None:
        return []
    try:
        result = subprocess.run(
            [ip, "-j", "-4", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [
        address.model_copy(update={"link_mbps": link_speed(address.interface, sys_net)})
        for address in parse_ip_addresses(result.stdout)
    ]


def probe_system(
    *,
    sample_interval: float = 0.2,
    proc: Path = PROC,
    os_release: Path = OS_RELEASE,
    sys_net: Path = SYS_NET,
) -> SystemInfo:
    """Describe this node. Never raises.

    CPU use is sampled over ``sample_interval`` rather than read as a
    since-boot average, which would barely move on a long-running node. The
    caller should run this off the event loop: it sleeps for the sample.
    """
    first = parse_cpu_times(_read(proc / "stat") or "")
    if first is not None:
        time.sleep(sample_interval)
    second = parse_cpu_times(_read(proc / "stat") or "")

    return SystemInfo(
        hostname=socket.gethostname(),
        os=parse_os_release(_read(os_release) or ""),
        kernel=platform.release() or None,
        cpu_model=parse_cpu_model(_read(proc / "cpuinfo") or ""),
        cpu_threads=os.cpu_count() or 0,
        cpu_pct=cpu_percent(first, second) if first and second else None,
        load_1m=parse_loadavg(_read(proc / "loadavg") or ""),
        uptime_s=parse_uptime(_read(proc / "uptime") or ""),
        addresses=_addresses(sys_net),
    )


__all__ = [
    "NetAddress",
    "SystemInfo",
    "cpu_percent",
    "link_speed",
    "parse_cpu_model",
    "parse_cpu_times",
    "parse_ip_addresses",
    "parse_loadavg",
    "parse_os_release",
    "parse_uptime",
    "probe_system",
]
