"""The interactive terminal dashboard: `huddle tui`.

Rendering only reuses the poller's snapshot; every control (start/stop/switch
model) is a thin call into `ClusterPoller`'s wrappers over the existing
`/cluster/*` endpoints. No placement, peer-resolution, or orchestration logic
lives here — see `poller.py` for where the data actually comes from.
"""

from __future__ import annotations

from typing import ClassVar

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Label, RichLog, Select, Static

from huddle.config import HuddleConfig
from huddle.tui.poller import ClusterPoller, ClusterSnapshot, NodeSnapshot

_STATE_COLOR = {"running": "green", "starting": "yellow", "degraded": "red", "stopped": "dim"}


def _cluster_state(cluster: ClusterSnapshot) -> str:
    status = cluster.cluster
    if status is None:
        return "stopped"
    if status.starting:
        return "starting"
    if status.running:
        return "running"
    if status.degraded:
        return "degraded"
    return "stopped"


def _render_node_text(node: NodeSnapshot) -> str:
    lines = [f"[b]{node.name}[/b]  ({node.role})"]
    if not node.reachable:
        lines.append("[red]unreachable[/red]")
        if node.error:
            lines.append(f"[dim]{node.error}[/dim]")
        return "\n".join(lines)

    hardware = node.hardware
    if hardware is not None:
        used = hardware.ram_total_mib - hardware.ram_available_mib
        pct = (used / hardware.ram_total_mib * 100) if hardware.ram_total_mib else 0.0
        lines.append(f"RAM  {used:,}/{hardware.ram_total_mib:,} MiB ({pct:.0f}%)")
        for device in hardware.devices:
            free = device.free_mib if device.free_mib is not None else device.total_mib
            total = device.total_mib
            mem = f"{free:,}/{total:,} MiB free" if free is not None and total is not None else "-"
            extra = ""
            if device.util_pct is not None:
                power = f"{device.power_w:.0f}W" if device.power_w is not None else "-W"
                extra = f"  {device.util_pct}%  {device.temp_c}C  {power}"
            lines.append(f"  {device.id}: {device.name[:24]}  {mem}{extra}")
    if node.role == "head" and node.backend is not None and node.backend.tokens_per_sec is not None:
        lines.append(f"tok/s: {node.backend.tokens_per_sec:.1f}")
    lines.append(f"layers: {node.layers}")
    return "\n".join(lines)


def _detail(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
    except ValueError:
        pass
    return exc.response.text or str(exc)


class ConfirmStopScreen(ModalScreen[bool]):
    """Confirm before stopping a running cluster."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Stop the cluster?"),
            Horizontal(
                Button("Stop", id="confirm", variant="error"),
                Button("Cancel", id="cancel"),
            ),
            id="confirm-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class ModelSelectScreen(ModalScreen[str | None]):
    """Pick a model to switch to."""

    def __init__(self, models: list[str]) -> None:
        super().__init__()
        self._models = models

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Switch model"),
            Select[str]([(m, m) for m in self._models], id="model-choice"),
            Horizontal(
                Button("Load", id="confirm", variant="primary"),
                Button("Cancel", id="cancel"),
            ),
            id="model-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "confirm":
            self.dismiss(None)
            return
        value = self.query_one("#model-choice", Select).value
        self.dismiss(value if isinstance(value, str) else None)


class HuddleTUI(App[None]):
    """Monitor and control a Huddle cluster from the terminal."""

    CSS_PATH = "app.tcss"
    TITLE = "Huddle"
    BINDINGS: ClassVar = [
        ("s", "start_cluster", "Start"),
        ("x", "stop_cluster", "Stop"),
        ("m", "switch_model", "Switch model"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, config: HuddleConfig, *, poll_interval: float | None = None) -> None:
        super().__init__()
        self.config = config
        self.poll_interval = poll_interval or config.supervisor.watch_interval
        self.poller = ClusterPoller(config)
        self.snapshot: ClusterSnapshot | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(id="nodes")
        yield Static("connecting ...", id="topology")
        yield Horizontal(
            Static("", id="cluster-status", classes="panel"),
            Vertical(
                Select[str]([], id="model-select", prompt="switch model"),
                Static(
                    "sharding: automatic — Huddle splits layers across every "
                    "connected node; there is nothing to configure here",
                    id="launch-note",
                ),
                id="launch",
                classes="panel",
            ),
            id="status-row",
        )
        yield RichLog(id="log", max_lines=200, wrap=False)
        yield Footer()

    async def on_mount(self) -> None:
        self.set_interval(self.poll_interval, self._schedule_refresh)
        await self.refresh_data()

    async def on_unmount(self) -> None:
        await self.poller.aclose()

    def _schedule_refresh(self) -> None:
        """Run a poll, cancelling any earlier one still in flight.

        Without this, a slow poll from an old timer tick can finish *after*
        a newer one (e.g. the refresh right after a start/stop action) and
        overwrite fresher data with stale data — the response order is not
        the request order. `exclusive=True` on a shared group makes each new
        refresh supersede whatever refresh came before it, so only the
        latest ever lands.
        """
        self.run_worker(self.refresh_data(), exclusive=True, group="refresh")

    async def refresh_data(self) -> None:
        try:
            snapshot = await self.poller.poll()
        except Exception as exc:
            self.notify(f"poll failed: {exc}", severity="error")
            return
        self.snapshot = snapshot
        await self._render_nodes(snapshot)
        self._render_topology(snapshot)
        self._render_cluster_status(snapshot)
        self._render_launch_panel(snapshot)
        await self._render_log()

    async def _render_nodes(self, snapshot: ClusterSnapshot) -> None:
        container = self.query_one("#nodes", Horizontal)
        await container.remove_children()
        for node in snapshot.nodes:
            classes = "node-box" if node.reachable else "node-box unreachable"
            await container.mount(Static(_render_node_text(node), classes=classes))

    def _render_topology(self, snapshot: ClusterSnapshot) -> None:
        topology = self.query_one("#topology", Static)
        if snapshot.coordinator_error:
            topology.update(f"[yellow]{snapshot.coordinator_error}[/yellow]")
            return
        if not snapshot.nodes:
            topology.update("[dim]no nodes[/dim]")
            return
        parts = [f"[{n.name}: {n.role}, {n.layers} layers]" for n in snapshot.nodes]
        topology.update("  ──rpc──▶  ".join(parts))

    def _render_cluster_status(self, snapshot: ClusterSnapshot) -> None:
        panel = self.query_one("#cluster-status", Static)
        cluster = snapshot.cluster
        if cluster is None:
            panel.update("[dim]no cluster status available[/dim]")
            return
        state = _cluster_state(snapshot)
        color = _STATE_COLOR[state]
        lines = [
            f"model: {cluster.model or '-'}",
            f"status: [{color}]{state}[/{color}]   "
            f"restarts: {cluster.restarts}   supervising: {cluster.supervising}",
        ]
        if cluster.last_failure:
            lines.append(f"[red]last failure: {cluster.last_failure}[/red]")
        panel.update("\n".join(lines))

    def _render_launch_panel(self, snapshot: ClusterSnapshot) -> None:
        select = self.query_one("#model-select", Select)
        select.set_options([(name, name) for name in snapshot.available_models])
        if snapshot.loaded_model and snapshot.loaded_model in snapshot.available_models:
            select.value = snapshot.loaded_model

    async def _render_log(self) -> None:
        log_widget = self.query_one("#log", RichLog)
        lines = await self.poller.head_logs()
        log_widget.clear()
        for line in lines[-200:]:
            log_widget.write(line)

    # -- actions -----------------------------------------------------------

    def action_refresh(self) -> None:
        self._schedule_refresh()

    def action_start_cluster(self) -> None:
        self.run_worker(self._start_cluster())

    async def _start_cluster(self) -> None:
        try:
            await self.poller.start_cluster()
        except httpx.HTTPStatusError as exc:
            self.notify(f"start failed: {_detail(exc)}", severity="error")
        except httpx.HTTPError as exc:
            self.notify(f"start failed: {exc}", severity="error")
        self._schedule_refresh()

    def action_stop_cluster(self) -> None:
        self.push_screen(ConfirmStopScreen(), self._on_stop_confirmed)

    def _on_stop_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.run_worker(self._stop_cluster())

    async def _stop_cluster(self) -> None:
        try:
            await self.poller.stop_cluster()
        except httpx.HTTPError as exc:
            self.notify(f"stop failed: {exc}", severity="error")
        self._schedule_refresh()

    def action_switch_model(self) -> None:
        models = self.snapshot.available_models if self.snapshot else []
        if not models:
            self.notify("no models available", severity="warning")
            return
        self.push_screen(ModelSelectScreen(models), self._on_model_chosen)

    def _on_model_chosen(self, model: str | None) -> None:
        if model:
            self.run_worker(self._switch_model(model))

    async def _switch_model(self, model: str) -> None:
        try:
            await self.poller.switch_model(model)
        except httpx.HTTPStatusError as exc:
            self.notify(f"switch failed: {_detail(exc)}", severity="error")
        except httpx.HTTPError as exc:
            self.notify(f"switch failed: {exc}", severity="error")
        self._schedule_refresh()


def run_tui(config: HuddleConfig, *, poll_interval: float | None = None) -> None:
    HuddleTUI(config, poll_interval=poll_interval).run()


__all__ = ["HuddleTUI", "run_tui"]
