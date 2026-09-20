"""Pure rendering: a `ClusterSnapshot` in, a Rich renderable out.

No I/O and no Rich `Live`/`Console` state here, so every function is testable
by calling it and inspecting the result — the same reason `poller.py` has no
Textual/Rich import. `app.py` is the only place that owns a terminal.

Modeled on how exo's own terminal visualizer
(`exo/viz/topology_viz.py`, using `rich.live.Live` and hand-built ASCII/
Unicode layout) actually works, rather than a widget framework: build a
renderable from the current state, hand it to `Live.update()`, done. Huddle's
topology is always a single pipeline (head, then peers in RPC order) and
never an arbitrary mesh, so nodes are drawn left to right in that order
rather than exo's circular graph layout.
"""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from huddle.config import HuddleConfig
from huddle.doctor import api_base_url
from huddle.tui.poller import ClusterSnapshot, NodeSnapshot

_METER_WIDTH = 18
_LOG_LINES = 14


def cluster_state(snapshot: ClusterSnapshot) -> str:
    status = snapshot.cluster
    if status is None:
        return "stopped"
    if status.starting:
        return "starting"
    if status.running:
        return "running"
    if status.degraded:
        return "degraded"
    return "stopped"


_STATE_STYLE = {"running": "green", "starting": "yellow3", "degraded": "red", "stopped": "grey50"}


def meter(fraction: float | None, *, width: int = _METER_WIDTH) -> Text:
    """A colored bar-graph meter, e.g. `##########--------  42%`.

    Green/yellow/red thresholds match the common htop-style convention:
    comfortable, busy, saturated. `None` (no data — e.g. a non-NVIDIA device)
    renders as a dim, unfilled placeholder rather than a fabricated number.
    """
    if fraction is None:
        text = Text("·" * width, style="grey35")
        text.append("  n/a", style="grey35")
        return text
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    color = "green" if fraction < 0.6 else ("yellow3" if fraction < 0.85 else "red")
    text = Text("█" * filled, style=color)
    text.append("░" * (width - filled), style="grey35")
    text.append(f"  {fraction * 100:>3.0f}%")
    return text


def _labeled_meter(label: str, fraction: float | None) -> Table:
    row = Table.grid(padding=(0, 1))
    row.add_column(width=4)
    row.add_column()
    row.add_row(label, meter(fraction))
    return row


def node_panel(node: NodeSnapshot) -> Panel:
    body = Table.grid(padding=(0, 0))
    body.add_column()
    title = f"{node.name} ({node.role})"

    if not node.reachable:
        body.add_row(Text("unreachable", style="bold red"))
        if node.error:
            body.add_row(Text(node.error, style="grey50"))
        return Panel(body, title=title, border_style="red")

    hardware = node.hardware
    if hardware is not None:
        used = hardware.ram_total_mib - hardware.ram_available_mib
        ram_fraction = used / hardware.ram_total_mib if hardware.ram_total_mib else None
        body.add_row(_labeled_meter("ram", ram_fraction))
        body.add_row(Text(f"{used:,}/{hardware.ram_total_mib:,} MiB", style="grey50"))
        for device in hardware.devices:
            total, free = device.total_mib, device.free_mib
            used_frac = (total - free) / total if total and free is not None else None
            body.add_row(Text(f"{device.id}  {device.name}", style="bold"))
            body.add_row(_labeled_meter("mem", used_frac))
            if device.util_pct is not None:
                body.add_row(_labeled_meter("gpu", device.util_pct / 100))
                extra = f"{device.temp_c}C" if device.temp_c is not None else "-C"
                if device.power_w is not None:
                    extra += f"  {device.power_w:.0f}W"
                body.add_row(Text(extra, style="grey50"))

    if node.role == "head" and node.backend is not None and node.backend.tokens_per_sec is not None:
        body.add_row(Text(f"{node.backend.tokens_per_sec:.1f} tok/s", style="bold yellow"))
    body.add_row(Text(f"layers: {node.layers}", style="grey50"))
    return Panel(body, title=title, border_style="green")


def node_row(snapshot: ClusterSnapshot) -> RenderableType:
    if not snapshot.nodes:
        return Text("no nodes", style="grey50")
    return Columns([node_panel(n) for n in snapshot.nodes], equal=True, expand=True)


def pipeline_line(snapshot: ClusterSnapshot) -> Text:
    """A left-to-right chain, e.g. `omarchy (25L) -- rpc --> maksood (34L)`.

    Deliberately linear, not a graph: Huddle only ever has one head plus N
    peers in a fixed RPC order, so a mesh layout (exo's circular placement)
    would draw a topology Huddle can never actually have.
    """
    if snapshot.coordinator_error:
        return Text(snapshot.coordinator_error, style="yellow3")
    if not snapshot.nodes:
        return Text("connecting ...", style="grey50")
    text = Text(justify="center")
    for i, node in enumerate(snapshot.nodes):
        if i:
            text.append("  -- rpc --> ", style="grey35")
        text.append("● ", style="green" if node.reachable else "red")
        text.append(f"{node.name} ")
        text.append(f"({node.layers}L)", style="grey50")
    return text


def cluster_panel(snapshot: ClusterSnapshot) -> Panel:
    cluster = snapshot.cluster
    if cluster is None:
        return Panel(Text("no cluster status available", style="grey50"), title="cluster")

    state = cluster_state(snapshot)
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text(f"model  {cluster.model or '-'}"))
    row = Table.grid(padding=(0, 1))
    row.add_row("status", Text(state, style=f"bold {_STATE_STYLE[state]}"))
    body.add_row(row)
    body.add_row(
        Text(f"restarts: {cluster.restarts}   supervising: {cluster.supervising}", style="grey50")
    )
    if cluster.last_failure:
        body.add_row(Text(f"last failure: {cluster.last_failure}", style="red"))
    return Panel(body, title="cluster", border_style="green")


def launch_panel(snapshot: ClusterSnapshot) -> Panel:
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text(f"loaded: {snapshot.loaded_model or '-'}"))
    others = [m for m in snapshot.available_models if m != snapshot.loaded_model]
    if others:
        shown = ", ".join(others[:4]) + (" ..." if len(others) > 4 else "")
        body.add_row(Text(f"available: {shown}", style="grey50"))
    body.add_row(Text("sharding: automatic (pipeline split, every connected node)", style="grey50"))
    return Panel(body, title="model", border_style="green")


def status_row(snapshot: ClusterSnapshot) -> RenderableType:
    return Columns([cluster_panel(snapshot), launch_panel(snapshot)], equal=True, expand=True)


def banner(snapshot: ClusterSnapshot | None, config: HuddleConfig) -> Panel:
    node_count = len(snapshot.nodes) if snapshot else 0
    plural = "node" if node_count == 1 else "nodes"
    endpoint = f"{api_base_url(config)}/v1/chat/completions"
    text = Text(justify="center")
    text.append("H U D D L E\n", style="bold green")
    text.append(f"cluster ({node_count} {plural})\n", style="grey50")
    text.append(f"api: {endpoint}", style="grey50")
    return Panel(text, border_style="green")


def log_panel(lines: list[str], *, height: int = _LOG_LINES) -> Panel:
    tail = lines[-height:]
    content = Text("\n".join(tail) if tail else "(no output yet)", style="grey70", no_wrap=True)
    return Panel(content, title="log", border_style="green", height=height + 2)


def footer(message: str | None = None) -> Text:
    base = Text("s Start   x Stop   m Switch model   r Refresh   q Quit", style="grey50")
    if message:
        base.append("   |   ")
        base.append(message, style="yellow3")
    return base


def frame(
    snapshot: ClusterSnapshot | None, config: HuddleConfig, *, message: str | None = None
) -> Group:
    """The whole screen for one refresh, stacked top to bottom.

    A fixed-height log panel rather than "fill remaining space": simpler and
    more predictable than fighting Rich's `Layout` sizing for a
    variable-height top section, and it's exactly what exo's own visualizer
    does for its download/prompt panels (fixed `size=15`/`size=25`).
    """
    parts: list[RenderableType] = [banner(snapshot, config)]
    if snapshot is not None:
        parts += [node_row(snapshot), pipeline_line(snapshot), status_row(snapshot)]
        parts.append(log_panel(snapshot.log_lines))
    else:
        parts.append(Text("connecting ...", style="grey50"))
    parts.append(footer(message))
    return Group(*parts)


__all__ = [
    "banner",
    "cluster_panel",
    "cluster_state",
    "footer",
    "frame",
    "launch_panel",
    "log_panel",
    "meter",
    "node_panel",
    "node_row",
    "pipeline_line",
    "status_row",
]
