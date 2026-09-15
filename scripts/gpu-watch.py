#!/usr/bin/env python3
"""Watch GPU usage across the cluster.

Answers one question: is Huddle actually using these GPUs? Desktop applications
hold hundreds of MiB on a workstation GPU, so a raw "memory used" number cannot
tell you whether inference is running. This separates memory held by llama.cpp
processes from everything else.

Usage:
    scripts/gpu-watch.py                  # watch omarchy and maksood
    scripts/gpu-watch.py --once           # print once and exit
    scripts/gpu-watch.py -i 5 nodeA nodeB # every 5s, specific hosts

Hosts are SSH targets: aliases from ~/.ssh/config, or user@host.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

DEFAULT_HOSTS = ["omarchy", "maksood"]

# Processes that mean "Huddle is using this GPU", as opposed to a browser.
HUDDLE_PROCESSES = ("llama-server", "ggml-rpc-server", "rpc-server", "llama-cli")

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"

REMOTE = r"""
command -v nvidia-smi >/dev/null || { echo "NOGPU"; exit 0; }
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
           --format=csv,noheader,nounits | sed 's/^/GPU|/'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv,noheader,nounits 2>/dev/null | sed 's/^/PROC|/'
"""


@dataclass
class Gpu:
    name: str
    used_mib: int
    total_mib: int
    util: int
    temp: int


@dataclass
class NodeState:
    host: str
    gpus: list[Gpu] = field(default_factory=list)
    huddle_mib: int = 0
    other_mib: int = 0
    huddle_procs: list[tuple[str, int]] = field(default_factory=list)
    error: str | None = None


def probe(host: str, timeout: float = 15.0) -> NodeState:
    """Collect GPU state from one host. Never raises."""
    state = NodeState(host=host)
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout)}", host, REMOTE],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        state.error = f"{type(exc).__name__}"
        return state

    if result.returncode != 0:
        state.error = (result.stderr.strip().splitlines() or ["unreachable"])[-1][:60]
        return state
    if "NOGPU" in result.stdout:
        state.error = "no nvidia-smi"
        return state

    for line in result.stdout.splitlines():
        kind, _, payload = line.partition("|")
        fields = [f.strip() for f in payload.split(",")]
        try:
            if kind == "GPU" and len(fields) >= 5:
                state.gpus.append(
                    Gpu(fields[0], int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]))
                )
            elif kind == "PROC" and len(fields) >= 3:
                name, mib = fields[1], int(fields[2])
                if any(marker in name for marker in HUDDLE_PROCESSES):
                    state.huddle_mib += mib
                    state.huddle_procs.append((name.rsplit("/", 1)[-1], mib))
                else:
                    state.other_mib += mib
        except ValueError:
            continue
    return state


def bar(fraction: float, width: int = 18) -> str:
    filled = max(0, min(width, round(fraction * width)))
    colour = GREEN if fraction < 0.7 else (YELLOW if fraction < 0.9 else RED)
    return f"{colour}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def render(states: list[NodeState]) -> str:
    out = [f"{BOLD}huddle gpu{RESET}  {DIM}{time.strftime('%H:%M:%S')}{RESET}", "─" * 74]
    cluster_huddle = 0

    for state in states:
        if state.error:
            out.append(f"{BOLD}{state.host:10}{RESET} {RED}{state.error}{RESET}")
            continue

        for gpu in state.gpus:
            fraction = gpu.used_mib / gpu.total_mib if gpu.total_mib else 0.0
            out.append(
                f"{BOLD}{state.host:10}{RESET}{gpu.name[:22]:23}"
                f"{gpu.used_mib:>6}/{gpu.total_mib} MiB  {bar(fraction)}"
                f" {gpu.util:>3}%  {gpu.temp:>2}C"
            )

        cluster_huddle += state.huddle_mib
        if state.huddle_procs:
            for name, mib in sorted(state.huddle_procs, key=lambda p: -p[1]):
                out.append(
                    f"{' ' * 10}{CYAN}└ {name:<28}{RESET}{mib:>6} MiB  {DIM}inference{RESET}"
                )
        else:
            out.append(f"{' ' * 10}{DIM}└ no llama.cpp process on this GPU{RESET}")
        if state.other_mib:
            out.append(f"{' ' * 10}{DIM}└ other apps{'':18}{state.other_mib:>6} MiB{RESET}")

    out.append("─" * 74)
    verdict = (
        f"{GREEN}inference using {cluster_huddle} MiB across the cluster{RESET}"
        if cluster_huddle
        else f"{YELLOW}no llama.cpp process is holding GPU memory anywhere{RESET}"
    )
    out.append(f"  {verdict}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hosts", nargs="*", default=DEFAULT_HOSTS, help="SSH targets")
    parser.add_argument(
        "-i", "--interval", type=float, default=2.0, help="seconds between refreshes"
    )
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()
    hosts = args.hosts or DEFAULT_HOSTS

    try:
        while True:
            with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
                states = list(pool.map(probe, hosts))
            frame = render(states)
            if args.once:
                print(frame)
                return 0
            # Home the cursor and clear below, so the display updates in place
            # without the flicker of a full clear.
            sys.stdout.write("\033[H\033[J" + frame + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
