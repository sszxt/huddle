"""Bringing the whole cluster up and down.

The sequence matters. Workers must be listening before the head starts, because
the head connects to every RPC endpoint while loading and streams each remote
layer's weights across on the way. And if the head fails to come up, workers we
started have to be torn down again, or the next attempt hits ports already held.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel

from huddle.agent.service import BackendService
from huddle.config import HuddleConfig, PeerConfig
from huddle.coordinator.cluster import PeerReport, build_device_list, query_peers
from huddle.coordinator.planner import PlacementDevice, Plan, plan_placement
from huddle.gguf import ModelInfo, read_gguf
from huddle.hardware import probe
from huddle.process import ProcessError

log = logging.getLogger("huddle.cluster")


class DevicePlacement(BaseModel):
    """One device and what it was given."""

    id: str
    name: str
    node: str
    layers: int
    free_mib: int
    skipped: str | None = None


class ClusterPlan(BaseModel):
    """A placement decision, in a form the API can return."""

    model: str
    n_layers: int
    n_gpu_layers: int
    per_layer_mib: float
    tensor_split: list[float]
    rpc_endpoints: list[str]
    placement: list[DevicePlacement]
    unreachable: list[str] = []
    warnings: list[str] = []


class ClusterStatus(BaseModel):
    """What the cluster is currently doing."""

    running: bool
    model: str | None = None
    head_node: str
    workers: list[str] = []
    plan: ClusterPlan | None = None


class ClusterService:
    """Plans a placement, then starts the processes that realise it."""

    def __init__(self, config: HuddleConfig, backend: BackendService) -> None:
        self.config = config
        self.backend = backend
        self._plan: ClusterPlan | None = None
        self._started_workers: list[PeerConfig] = []
        self._lock = asyncio.Lock()

    def status(self) -> ClusterStatus:
        return ClusterStatus(
            running=self.backend.running,
            model=self.backend.status().model,
            head_node=self.config.node.name,
            workers=[peer.name for peer in self._started_workers],
            plan=self._plan,
        )

    async def build_plan(
        self, model: str | None = None, *, n_ctx: int | None = None
    ) -> ClusterPlan:
        """Work out where layers should go, without starting anything."""
        info: ModelInfo = read_gguf(self.config.models.resolve(model))
        local = probe(self.config.node.name, self.config.binaries.llama_server)
        reports = await query_peers(self.config)

        ctx = n_ctx or self.config.backend.ctx_size
        reachable = [report for report in reports if report.reachable]

        devices, endpoints = build_device_list(local, reachable)
        if not devices:
            raise ProcessError("no usable devices found on this node or any peer")
        plan = self._plan_for(info, devices, ctx)

        # Drop peers that were given no layers and plan again. Every endpoint in
        # --rpc is one llama.cpp connects to at load and depends on afterwards,
        # so a peer holding nothing is pure liability: it can still take the head
        # down with it, and buys nothing. Re-planning is required rather than
        # cosmetic, because removing a peer removes its device slots and shifts
        # every later --tensor-split position.
        used = {
            device.node
            for device, layers in zip(plan.devices, plan.layers_per_device, strict=True)
            if layers > 0
        }
        active = [report for report in reachable if report.peer.name in used]
        if len(active) != len(reachable):
            devices, endpoints = build_device_list(local, active)
            plan = self._plan_for(info, devices, ctx)

        return _to_cluster_plan(info, plan, endpoints, reports)

    def _plan_for(self, info: ModelInfo, devices: list[PlacementDevice], n_ctx: int) -> Plan:
        return plan_placement(
            info,
            devices,
            n_ctx=n_ctx,
            headroom=self.config.planner.headroom,
            use_unified_memory=self.config.planner.use_unified_memory,
        )

    async def start(self, model: str | None = None) -> ClusterStatus:
        """Start workers, then the head, and roll back if the head fails."""
        async with self._lock:
            if self.backend.running:
                raise ProcessError("cluster is already running; stop it first")

            plan = await self.build_plan(model)
            for warning in plan.warnings:
                log.warning("%s", warning)

            # Only peers actually carrying layers are worth starting.
            wanted = _peers_with_layers(self.config.peers, plan)
            await self._start_workers(wanted)

            try:
                await self.backend.start(
                    model,
                    rpc_endpoints=plan.rpc_endpoints or None,
                    tensor_split=plan.tensor_split or None,
                    n_gpu_layers=plan.n_gpu_layers,
                )
            except Exception:
                # Leaving workers up would hold their ports and confuse the next
                # attempt into thinking a stale cluster is healthy.
                await self._stop_workers()
                raise

            self._plan = plan
            return self.status()

    async def stop(self) -> ClusterStatus:
        """Stop the head first, then the workers it depends on."""
        async with self._lock:
            await self.backend.stop()
            await self._stop_workers()
            self._plan = None
            return self.status()

    async def _start_workers(self, peers: list[PeerConfig]) -> None:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for peer in peers:
                url = f"http://{peer.host}:{peer.agent_port}/agent/rpc/start"
                try:
                    response = await client.post(url)
                    # 409 means it is already running, which is what we want.
                    if response.status_code not in (200, 409):
                        raise ProcessError(
                            f"peer {peer.name} refused to start its worker: "
                            f"{response.status_code} {response.text[:200]}"
                        )
                except httpx.HTTPError as exc:
                    await self._stop_workers()
                    raise ProcessError(f"peer {peer.name} unreachable: {exc}") from exc
                self._started_workers.append(peer)

    async def _stop_workers(self) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for peer in self._started_workers:
                url = f"http://{peer.host}:{peer.agent_port}/agent/rpc/stop"
                try:
                    await client.post(url)
                except httpx.HTTPError as exc:
                    # Worth logging but not worth failing a shutdown over.
                    log.warning("could not stop worker on %s: %s", peer.name, exc)
        self._started_workers = []


def _peers_with_layers(peers: list[PeerConfig], plan: ClusterPlan) -> list[PeerConfig]:
    """Peers whose endpoints appear in ``--rpc``.

    Every one of them must have a live worker: llama.cpp connects to each
    endpoint while loading, and an endpoint with nothing behind it fails the
    whole start. The plan has already dropped peers holding no layers, so this
    is exactly the set that needs starting.
    """
    wanted = set(plan.rpc_endpoints)
    return [peer for peer in peers if peer.rpc_endpoint in wanted]


def _to_cluster_plan(
    info: ModelInfo, plan: Plan, endpoints: list[str], reports: list[PeerReport]
) -> ClusterPlan:
    skipped = dict(plan.excluded)
    placement = [
        DevicePlacement(
            id=device.id,
            name=device.name,
            node=device.node,
            layers=layers,
            free_mib=device.free_mib,
            skipped=skipped.get(device.id),
        )
        for device, layers in zip(plan.devices, plan.layers_per_device, strict=True)
    ]
    return ClusterPlan(
        model=info.path.name,
        n_layers=info.n_layers,
        n_gpu_layers=plan.n_gpu_layers,
        per_layer_mib=round(plan.per_layer_mib, 1),
        tensor_split=plan.tensor_split,
        rpc_endpoints=endpoints,
        placement=placement,
        unreachable=[r.peer.name for r in reports if not r.reachable],
        warnings=plan.warnings,
    )


__all__ = ["ClusterPlan", "ClusterService", "ClusterStatus", "DevicePlacement", "PlacementDevice"]
